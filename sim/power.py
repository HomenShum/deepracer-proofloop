"""
Measure the instrument before trusting it.

The experiment compared designs separated by at most 0.35 s using a measurement
whose standard deviation was 0.65 to 0.80 s. The noise was twice the prize, so
no method could be distinguished. That is an underpowered experiment, not a
result about agents.

This script asks the only question that matters first:

    How much inner training budget is needed before the noise is small enough
    to see a real difference?

`opt10` already reaches sd 0.123 at the current budget, so low variance is
reachable. If sd falls to about 0.12 for every design, the 0.35 s headroom
becomes detectable with a handful of seeds instead of sixty.

    python sim/power.py --seeds 9
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "scorer"))

import train as trainer  # noqa: E402

PHYSICAL_FLOOR = 18.147
CONTENDERS = {
    "opt9": ROOT / "data" / "reward_functions" / "reward_function_type3_opt9.py",
    "opt10": ROOT / "data" / "reward_functions" / "reward_function_type3_opt10.py",
}

# Increasing inner training budget. The question is where the variance settles.
BUDGETS = [(8, 32), (14, 48), (22, 64), (30, 96)]


def seeds_needed(sd: float, effect: float) -> float:
    """Rough seeds per design to detect `effect` at about 80 percent power."""
    if effect <= 0:
        return float("inf")
    return 16.0 * (sd ** 2) / (effect ** 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=9)
    ap.add_argument("--out", default="sim/power_results.json")
    args = ap.parse_args()

    print("\nVariance against inner training budget")
    print(f"physical floor {PHYSICAL_FLOOR:.3f}s, {args.seeds} seeds per cell\n")
    print(f"  {'design':<7} {'iters':>6} {'pop':>5} {'mean':>8} {'sd':>7} "
          f"{'DNF':>4} {'seeds for 0.35s':>16} {'for 0.10s':>11}")
    print(f"  {'-'*7} {'-'*6} {'-'*5} {'-'*8} {'-'*7} {'-'*4} {'-'*16} {'-'*11}")

    rows = []
    for iters, pop in BUDGETS:
        for label, path in CONTENDERS.items():
            laps, dnf = [], 0
            t0 = time.time()
            for s in range(args.seeds):
                try:
                    r = trainer.train(path, iters=iters, pop=pop, seed=s)
                except Exception as e:
                    print(f"  {label:<7} {iters:>6} {pop:>5}  error: {type(e).__name__}")
                    break
                if r["lap_time"]:
                    laps.append(r["lap_time"])
                else:
                    dnf += 1
            if not laps:
                continue
            mean = statistics.fmean(laps)
            sd = statistics.pstdev(laps) if len(laps) > 1 else 0.0
            n35 = seeds_needed(sd, 0.35)
            n10 = seeds_needed(sd, 0.10)
            rows.append({"design": label, "iters": iters, "pop": pop, "mean": mean,
                         "sd": sd, "dnf": dnf, "elapsed_s": time.time() - t0,
                         "laps": laps})
            print(f"  {label:<7} {iters:>6} {pop:>5} {mean:>7.3f}s {sd:>7.3f} "
                  f"{dnf:>4} {n35:>16.0f} {n10:>11.0f}")

    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n  written to {args.out}")

    best = min((r for r in rows), key=lambda r: r["sd"], default=None)
    if best:
        print(f"\n  lowest variance: {best['design']} at "
              f"{best['iters']}x{best['pop']}, sd {best['sd']:.3f}")
        print(f"  at that sd, detecting the 0.35s headroom needs about "
              f"{seeds_needed(best['sd'], 0.35):.0f} seeds per design")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
