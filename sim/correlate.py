"""
Does maximising this reward function actually make the car faster?

    python sim/correlate.py --all
    python sim/correlate.py --target data/reward_functions/x.py

Method
    1. Simulate a grid of driving policies on the racing line. Each one produces
       a real lap time, or fails to finish.
    2. Score every finished lap with the reward function under test.
    3. Spearman rank correlation between mean reward and SPEED (negated lap
       time). A reward function that does its job scores rho close to +1.0.
    4. Also report whether the reward function's own top-ranked lap is the
       fastest lap, and how much slower it is when it is not.

The second measure is the one that matters in practice. Training maximises the
reward, so the lap the reward likes best is the lap you get.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scorer"))
sys.path.insert(0, str(ROOT / "sim"))

from core import load_reward_function  # noqa: E402
from track_sim import (  # noqa: E402
    load_line, load_actions, policy_grid, simulate, score_lap, spearman,
)

LINE = ROOT / "data" / "lines" / "optimals_newest_Ross_racing_line.txt"
ACTIONS = ROOT / "data" / "lines" / "AS21_newest_Ross_racing_line.txt"


def build_laps():
    line, actions = load_line(LINE), load_actions(ACTIONS)
    laps = [simulate(line, actions, p) for p in policy_grid()]
    return [l for l in laps if l.finished], len(laps)


def analyse(path: Path, laps):
    fn = load_reward_function(path)
    rewards = [score_lap(fn, l) for l in laps]
    times = [l.lap_time for l in laps]
    speeds = [-t for t in times]          # faster = higher

    rho = spearman(rewards, speeds)
    best_by_reward = max(range(len(laps)), key=lambda i: rewards[i])
    best_by_clock = min(range(len(laps)), key=lambda i: times[i])
    chosen, fastest = times[best_by_reward], times[best_by_clock]
    return {
        "name": path.name,
        "rho": rho,
        "chosen_time": chosen,
        "fastest_time": fastest,
        "penalty_s": chosen - fastest,
        "penalty_pct": (chosen - fastest) / fastest * 100.0 if fastest else 0.0,
        "chosen_policy": laps[best_by_reward].policy.name,
        "fastest_policy": laps[best_by_clock].policy.name,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    laps, attempted = build_laps()
    print(f"\nSimulated {attempted} policies on the Ross racing line. "
          f"{len(laps)} completed a lap.")
    if laps:
        ts = sorted(l.lap_time for l in laps)
        print(f"Lap times range {ts[0]:.2f}s to {ts[-1]:.2f}s. "
              f"A reward function has to prefer the low end.\n")

    paths = (sorted((ROOT / "data" / "reward_functions").glob("*.py"))
             if args.all else [Path(args.target)])

    rows = []
    for p in paths:
        try:
            rows.append(analyse(p, laps))
        except Exception as e:
            print(f"  could not analyse {p.name}: {e}")

    rows.sort(key=lambda r: -r["rho"])
    print(f"  {'reward function':<48} {'rho':>7} {'picks':>8} {'fastest':>8} {'cost':>8}")
    print(f"  {'-'*48} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
    for r in rows:
        print(f"  {r['name']:<48} {r['rho']:>7.3f} {r['chosen_time']:>7.2f}s "
              f"{r['fastest_time']:>7.2f}s {r['penalty_pct']:>6.1f}%")

    print("\n  rho      rank correlation between reward and speed. +1.0 is perfect.")
    print("  picks    lap time of the policy this reward function ranks highest.")
    print("  cost     how much slower that is than the fastest available lap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
