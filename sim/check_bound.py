"""
Why does a trained policy beat the stated physical floor?

The floor was recorded as 18.147 s, the sum of the racing line's own per-point
times. A trained policy reaches 17.311 s. One of two things is true:

  A. The bound is mislabelled. 18.147 s is the time to FOLLOW THAT LINE. The
     corridor allows a lateral offset, so a shorter path exists and the number
     is a reference time, not a floor.

  B. The simulator is still too permissive and the search is exploiting it.

This measures the path length the fast policy actually drives and compares it
with the racing line's own length. If the driven path is shorter, it is A.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "scorer"))

from track_sim import load_line, load_actions, MAX_LATERAL_ACCEL  # noqa: E402
from train import rollout, train                                   # noqa: E402

LINE = load_line(ROOT / "data" / "lines" / "optimals_newest_Ross_racing_line.txt")
ACTS = load_actions(ROOT / "data" / "lines" / "AS21_newest_Ross_racing_line.txt")
OPT9 = ROOT / "data" / "reward_functions" / "reward_function_type3_opt9.py"


def path_length(trace) -> float:
    return sum(math.dist((trace[i]["x"], trace[i]["y"]),
                         (trace[i + 1]["x"], trace[i + 1]["y"]))
               for i in range(len(trace) - 1))


def line_length(line) -> float:
    n = len(line)
    return sum(math.dist(line[i][:2], line[(i + 1) % n][:2]) for i in range(n))


def max_lateral_used(trace) -> float:
    """The highest lateral acceleration the driven path actually demanded."""
    worst = 0.0
    for i in range(1, len(trace) - 1):
        a = (trace[i - 1]["x"], trace[i - 1]["y"])
        b = (trace[i]["x"], trace[i]["y"])
        c = (trace[i + 1]["x"], trace[i + 1]["y"])
        A, B, C = math.dist(a, b), math.dist(b, c), math.dist(a, c)
        s = (A + B + C) / 2
        area = max(1e-12, s * (s - A) * (s - B) * (s - C)) ** 0.5
        if area < 1e-9:
            continue
        R = (A * B * C) / (4 * area)
        if R < 50:
            worst = max(worst, trace[i]["speed"] ** 2 / R)
    return worst


def main() -> int:
    print("\nTraining opt9 at the budget that produced 17.3 s ...")
    r = train(OPT9, iters=30, pop=96, seed=0)
    print(f"  lap {r['lap_time']:.3f}s   genome {r['final_genome']}")

    g = [r["final_genome"][k] for k in
         ("lookahead", "speed_scale", "lateral_bias", "curve_caution", "steer_gain")]
    fin, steps, lap, trace = rollout(LINE, ACTS, g)

    L_line = line_length(LINE)
    L_path = path_length(trace)
    lat = max_lateral_used(trace)

    print(f"\n  racing line length     {L_line:.3f} m")
    print(f"  path actually driven   {L_path:.3f} m")
    print(f"  difference             {L_path - L_line:+.3f} m "
          f"({(L_path - L_line) / L_line * 100:+.2f} percent)")
    print(f"\n  lateral accel cap in the model   {MAX_LATERAL_ACCEL:.3f} m/s^2")
    print(f"  highest lateral accel demanded   {lat:.3f} m/s^2")
    print(f"  lateral_bias the search chose    {g[2]:+.4f} m")

    print("\n  VERDICT")
    if L_path < L_line - 0.01:
        print("    The driven path is SHORTER than the racing line.")
        print("    18.147 s is a reference time for following that line, not a floor.")
        print("    The bound was mislabelled. Cutting inside the line is legal here.")
    elif lat > MAX_LATERAL_ACCEL + 1e-6:
        print("    The path demands more grip than the model allows.")
        print("    The simulator is still too permissive.")
    else:
        print("    The path is no shorter and within the grip cap.")
        print("    The speed profile itself is the remaining suspect.")
    return 0


if __name__ == "__main__":
    main()
