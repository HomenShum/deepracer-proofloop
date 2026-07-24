"""
Is the LLM's win real, or is it CEM noise?

The first run showed GLM 5.2 at 18.40s against the human's opt9 at 18.47s.
That is a 0.07s margin, 0.4 percent, produced by a stochastic optimiser from a
single seed. It is exactly the size of result that is usually noise.

This re-trains both across N seeds and reports the spread. If the intervals
overlap, there is no win to claim.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "search"))
sys.path.insert(0, str(ROOT / "sim"))

import train as trainer                     # noqa: E402
from evaluator import HUMAN_BEST, PHYSICAL_FLOOR  # noqa: E402

CONTENDERS = {
    "human opt9": ROOT / "data" / "reward_functions" / "reward_function_type3_opt9.py",
    "human opt10": ROOT / "data" / "reward_functions" / "reward_function_type3_opt10.py",
    "agent param search A": ROOT / "search" / "best_param_reward.py",
    "agent LLM (GLM 5.2)": ROOT / "search" / "best_llm_reward.py",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--pop", type=int, default=32)
    args = ap.parse_args()

    print(f"\nHead to head across {args.seeds} seeds "
          f"(inner CEM {args.iters} x {args.pop})")
    print(f"physical floor {PHYSICAL_FLOOR:.3f}s\n")
    print(f"  {'design':<24} {'mean':>8} {'best':>8} {'worst':>8} {'stdev':>8} {'DNF':>5}")
    print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")

    table = {}
    for label, path in CONTENDERS.items():
        if not path.exists():
            print(f"  {label:<24} missing: {path.name}")
            continue
        laps, dnf = [], 0
        for s in range(args.seeds):
            try:
                r = trainer.train(path, iters=args.iters, pop=args.pop, seed=s)
            except Exception as e:
                print(f"  {label:<24} training error: {type(e).__name__}")
                break
            if r["lap_time"]:
                laps.append(r["lap_time"])
            else:
                dnf += 1
        if not laps:
            print(f"  {label:<24} {'all DNF':>8}")
            continue
        table[label] = laps
        sd = statistics.pstdev(laps) if len(laps) > 1 else 0.0
        print(f"  {label:<24} {statistics.fmean(laps):>7.2f}s {min(laps):>7.2f}s "
              f"{max(laps):>7.2f}s {sd:>7.3f} {dnf:>5}")

    h = table.get("human opt9")
    a = table.get("agent LLM (GLM 5.2)")
    if h and a:
        mh, ma = statistics.fmean(h), statistics.fmean(a)
        sh = statistics.pstdev(h) if len(h) > 1 else 0.0
        sa = statistics.pstdev(a) if len(a) > 1 else 0.0
        print(f"\n  mean difference   {ma - mh:+.3f}s (agent minus human)")
        print(f"  human  range      {min(h):.2f} to {max(h):.2f}s")
        print(f"  agent  range      {min(a):.2f} to {max(a):.2f}s")
        overlap = not (max(a) < min(h) or max(h) < min(a))
        print(f"  ranges overlap    {overlap}")
        wins = sum(1 for x, y in zip(a, h) if x < y)
        print(f"  agent faster on   {wins} of {min(len(a), len(h))} matched seeds")
        print()
        if overlap or ma >= mh:
            print("  VERDICT: no win to claim. The margin is inside the noise.")
        else:
            print("  VERDICT: the agent is faster across every seed. The win holds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
