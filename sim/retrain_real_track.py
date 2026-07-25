"""Retrain on the real track, where off-track is measured from the centre line."""
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "scorer"))

import train as trainer  # noqa: E402

RACING_LINE_TIME = 23.362  # 2022_april_pro_ccw, sum of its own per-point times


def main():
    print(f"\nReal track: 2022_april_pro_ccw, width 1.067 m")
    print(f"Racing-line reference time {RACING_LINE_TIME:.3f} s")
    print("Off-track is now distance from the CENTRE line vs half width.\n")
    print(f"  {'design':<8} {'mean':>9} {'sd':>7} {'best':>9} {'DNF':>6}")
    print(f"  {'-'*8} {'-'*9} {'-'*7} {'-'*9} {'-'*6}")
    for label, f in [("opt9", "reward_function_type3_opt9.py"),
                     ("opt10", "reward_function_type3_opt10.py")]:
        laps, dnf = [], 0
        for s in range(5):
            try:
                r = trainer.train(ROOT / "data" / "reward_functions" / f,
                                  iters=22, pop=64, seed=s)
            except Exception as e:
                print(f"  {label:<8} error {type(e).__name__}: {e}")
                break
            if r["lap_time"]:
                laps.append(r["lap_time"])
            else:
                dnf += 1
        if laps:
            print(f"  {label:<8} {statistics.fmean(laps):>8.3f}s "
                  f"{statistics.pstdev(laps):>7.3f} {min(laps):>8.3f}s {dnf:>4}/5")
        else:
            print(f"  {label:<8} {'all DNF':>9} {'':>7} {'':>9} {dnf:>4}/5")

    print(f"\n  Any lap materially below {RACING_LINE_TIME:.3f}s now needs an "
          f"explanation,\n  because the racing line already uses the full track width.")


if __name__ == "__main__":
    main()
