"""
Train a policy to MAXIMISE a reward function, then time the resulting lap.

This is the experiment the correlation test only approximates. Correlation asks
whether a reward function ranks pre-made policies correctly. Training asks the
question that actually matters:

    If you optimise against this reward function, how fast is the car you get?

That is what happens during real training. The policy is whatever maximises the
reward. If the reward is wrong, you get a slower car, and no amount of training
fixes it.

Method: cross-entropy method (CEM). Sample policies from a Gaussian, simulate
each, keep the top fraction by TOTAL REWARD (never by lap time, which the
learner is not allowed to see), refit the Gaussian to the survivors, repeat.
The lap time is only revealed at the end, as the result.

Runs on CPU across all cores. No GPU, no AWS, no cost.

    python sim/train.py --target data/reward_functions/x.py
    python sim/train.py --all --iters 12 --pop 48
"""

from __future__ import annotations

import argparse
import io
import contextlib
import math
import os
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scorer"))
sys.path.insert(0, str(ROOT / "sim"))

from core import load_reward_function  # noqa: E402
from track import load_track  # noqa: E402
from track_sim import lateral_accel_from_line  # noqa: E402
from track_sim import (  # noqa: E402
    DT, MAX_STEPS, MAX_STEER_DEG, SPEED_TAU, WHEELBASE, _yaw_rate_deg,
    load_line, load_actions, _closest_index, _cross_track_error, _snap,
)

# Switched to the track for which REAL border geometry exists. The Ross line
# had no track file, so off-track could only be guessed from the racing line.
LINE_PATH = ROOT / "data" / "lines" / "optimals_newest_2022_april_pro_ccw.txt"
ACTION_PATH = ROOT / "data" / "lines" / "AS21_newest_2022_april_pro_ccw.txt"
TRACK_NAME = "2022_april_pro_ccw"

# Policy genome. Five knobs, wide enough that CEM can find genuinely
# different driving styles rather than nudging one.
#   0 lookahead        racing points ahead to aim at
#   1 speed_scale      fraction of the line's prescribed speed
#   2 lateral_bias     metres offset from the line
#   3 curve_caution    how much to slow for upcoming curvature
#   4 steer_gain       aggressiveness of the steering correction
GENOME = ["lookahead", "speed_scale", "lateral_bias", "curve_caution", "steer_gain"]
LO = [2.0, 0.30, -0.35, 0.0, 0.4]
HI = [22.0, 1.15, 0.35, 3.0, 2.0]
INIT_MU = [8.0, 0.70, 0.0, 1.0, 1.0]
INIT_SIGMA = [5.0, 0.22, 0.18, 0.9, 0.4]


def _clip(g):
    return [max(LO[i], min(HI[i], v)) for i, v in enumerate(g)]


def _curvature(line, i, span=6) -> float:
    n = len(line)
    a, b, c = line[(i - span) % n], line[i], line[(i + span) % n]
    h1 = math.atan2(b[1] - a[1], b[0] - a[0])
    h2 = math.atan2(c[1] - b[1], c[0] - b[0])
    d = abs(math.degrees(h2 - h1))
    return min(d, 360 - d) / 90.0


_TRACK = None
_AMAX = None


def _amax(line):
    global _AMAX
    if _AMAX is None:
        _AMAX = lateral_accel_from_line(line)
    return _AMAX


def _track():
    global _TRACK
    if _TRACK is None:
        _TRACK = load_track(TRACK_NAME)
    return _TRACK


def rollout(line, actions, g, track_width=None):
    """Simulate one genome. Returns (finished, steps, lap_time, params_trace)."""
    la = int(round(g[0]))
    speed_scale, lateral, caution, steer_gain = g[1], g[2], g[3], g[4]
    n = len(line)
    total_len = sum(math.hypot(line[(i + 1) % n][0] - line[i][0],
                               line[(i + 1) % n][1] - line[i][1]) for i in range(n))

    x, y = line[0][0], line[0][1]
    heading = math.degrees(math.atan2(line[1][1] - y, line[1][0] - x))
    speed = line[0][2] * speed_scale
    idx, visited = 0, 0
    trace = []

    for step in range(1, MAX_STEPS + 1):
        prev = idx
        idx = _closest_index(line, x, y, near=idx)
        visited += (idx - prev) % n

        t = line[(idx + la) % n]
        tp = line[(idx + la - 1) % n]
        tn = line[(idx + la + 1) % n]
        ndx, ndy = -(tn[1] - tp[1]), (tn[0] - tp[0])
        nl = math.hypot(ndx, ndy) or 1e-9
        tx, ty = t[0] + ndx / nl * lateral, t[1] + ndy / nl * lateral

        bearing = math.degrees(math.atan2(ty - y, tx - x))
        err = (bearing - heading + 180.0) % 360.0 - 180.0
        steer_cmd = max(-MAX_STEER_DEG, min(MAX_STEER_DEG, err * steer_gain))

        curv = _curvature(line, (idx + la) % n)
        speed_cmd = t[2] * speed_scale / (1.0 + caution * curv)
        steer, target = _snap(actions, steer_cmd, speed_cmd)

        # Real geometry. distance_from_center is measured from the CENTRE line
        # and compared with half the real track width, which is what DeepRacer
        # does. Measuring from the racing line let the car drive in the grass.
        tk = _track()
        cte = tk.distance_from_center(x, y)
        off = tk.is_offtrack(x, y, heading)          # point-in-polygon, per AWS
        all_on = tk.all_wheels_on_track(x, y, heading)

        trace.append({
            "all_wheels_on_track": all_on, "x": x, "y": y,
            "closest_waypoints": [idx, (idx + 1) % n],
            "distance_from_center": cte, "is_crashed": False,
            "is_left_of_center": tk.is_left_of_center(x, y), "is_offtrack": off, "is_reversed": False,
            "heading": heading, "progress": min(100.0, visited / n * 100.0),
            "speed": speed, "steering_angle": steer, "steps": step,
            "track_length": total_len, "track_width": tk.width,
            "waypoints": [[p[0], p[1]] for p in line],
        })

        if off:
            return False, step, float("inf"), trace
        if visited >= n:
            trace[-1]["progress"] = 100.0
            return True, step, step * DT, trace

        speed += (target - speed) * (DT / SPEED_TAU)
        heading += _yaw_rate_deg(speed, steer, _amax(line)) * DT
        heading = (heading + 180.0) % 360.0 - 180.0
        x += speed * math.cos(math.radians(heading)) * DT
        y += speed * math.sin(math.radians(heading)) * DT

    return False, MAX_STEPS, float("inf"), trace


_W = {}


def _init_worker(target: str):
    _W["line"] = load_line(LINE_PATH)
    _W["actions"] = load_actions(ACTION_PATH)
    _W["fn"] = load_reward_function(Path(target))


def _evaluate(g):
    """Total reward for one genome. The learner sees this and nothing else."""
    finished, steps, lap, trace = rollout(_W["line"], _W["actions"], g)
    total = 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        for p in trace:
            try:
                total += float(_W["fn"](dict(p)))
            except Exception:
                pass
    return total, finished, lap


def train(target: Path, iters=12, pop=48, elite_frac=0.25, seed=0, workers=None):
    random.seed(seed)
    mu, sigma = list(INIT_MU), list(INIT_SIGMA)
    n_elite = max(4, int(pop * elite_frac))
    workers = workers or min(os.cpu_count() or 4, 16)
    history = []

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker,
                             initargs=(str(target),)) as ex:
        for it in range(iters):
            pops = [_clip([random.gauss(mu[i], sigma[i]) for i in range(len(mu))])
                    for _ in range(pop)]
            results = list(ex.map(_evaluate, pops))
            ranked = sorted(zip(results, pops), key=lambda r: -r[0][0])
            elites = [g for _, g in ranked[:n_elite]]
            best_reward, best_finished, best_lap = ranked[0][0]

            for i in range(len(mu)):
                vals = [g[i] for g in elites]
                mu[i] = statistics.fmean(vals)
                sigma[i] = max(0.02, statistics.pstdev(vals) + 0.01)

            finished_laps = [r[2] for r, _ in ranked if r[1]]
            history.append({
                "iter": it + 1,
                "best_reward": best_reward,
                "best_lap": best_lap if best_finished else None,
                "finish_rate": len(finished_laps) / pop,
            })

        # Final policy = the distribution mean, evaluated once.
        final = list(ex.map(_evaluate, [_clip(mu)]))[0]

    return {
        "target": target.name,
        "final_genome": dict(zip(GENOME, [round(v, 4) for v in _clip(mu)])),
        "final_reward": final[0],
        "finished": final[1],
        "lap_time": final[2] if final[1] else None,
        "history": history,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = (sorted((ROOT / "data" / "reward_functions").glob("*.py"))
             if args.all else [Path(args.target)])

    print(f"\nTraining a policy against each reward function.")
    print(f"CEM: {args.iters} iterations x {args.pop} population, seed {args.seed}.")
    print("The learner optimises TOTAL REWARD only. It never sees lap time.\n")

    rows = []
    for p in paths:
        try:
            r = train(p, iters=args.iters, pop=args.pop, seed=args.seed)
        except Exception as e:
            print(f"  {p.name:<48} could not train: {e}")
            continue
        rows.append(r)
        lap = f"{r['lap_time']:.2f}s" if r["lap_time"] else "DNF"
        print(f"  {p.name:<48} -> {lap:>8}   (reward {r['final_reward']:.1f})")

    ok = [r for r in rows if r["lap_time"]]
    if ok:
        ok.sort(key=lambda r: r["lap_time"])
        print(f"\n{'='*74}\nRANKED BY THE LAP THE REWARD PRODUCED\n{'='*74}")
        print(f"  {'reward function':<48} {'lap':>9} {'vs best':>9}")
        best = ok[0]["lap_time"]
        for r in ok:
            print(f"  {r['target']:<48} {r['lap_time']:>8.2f}s "
                  f"{(r['lap_time']-best)/best*100:>8.1f}%")
        dnf = [r for r in rows if not r["lap_time"]]
        if dnf:
            print(f"\n  Did not finish: {', '.join(r['target'] for r in dnf)}")
            print("  A reward function whose optimum crashes the car is the worst outcome.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
