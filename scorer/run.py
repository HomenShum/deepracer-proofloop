"""
Command line runner for the DeepRacer reward function scorer.

    python scorer/run.py data/reward_functions/reward_function.py
    python scorer/run.py --all data/reward_functions
    python scorer/run.py --all data/reward_functions --json report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import evaluate, DEFAULT_TRACK_WIDTH  # noqa: E402


def print_report(rep) -> None:
    mark = "PASS" if rep.passed else "FAIL"
    print(f"\n{'='*74}")
    print(f"{mark}  {rep.function}   ({rep.track_points} racing points)")
    print(f"{'='*74}")
    print(f"  {'trajectory':<14} {'expect':<10} {'mean reward':>12} {'vs optimal':>12}  note")
    base = rep.optimal.mean if rep.optimal else 0.0
    for r in rep.results:
        ratio = "" if r.name == "optimal" else f"{(r.mean/base*100 if base else 0):>10.0f} %"
        err = f"  [{r.errors} errors]" if r.errors else ""
        print(f"  {r.name:<14} {r.expect:<10} {r.mean:>12.4f} {ratio:>12}  {r.note}{err}")
    if rep.failures:
        print("\n  FAILURES")
        for f in rep.failures:
            print(f"    - {f}")
    if rep.observations:
        print("\n  OBSERVATIONS (not gated)")
        for o in rep.observations:
            print(f"    - {o}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Score DeepRacer reward functions offline.")
    ap.add_argument("target", help="a reward function file, or a directory with --all")
    ap.add_argument("--all", action="store_true", help="score every .py file in the directory")
    ap.add_argument("--track-width", type=float, default=DEFAULT_TRACK_WIDTH)
    ap.add_argument("--json", help="write the full report to this path")
    args = ap.parse_args()

    target = Path(args.target)
    paths = sorted(target.glob("*.py")) if args.all else [target]
    if not paths:
        print("no reward function files found")
        return 2

    reports = []
    for p in paths:
        try:
            rep = evaluate(p, track_width=args.track_width)
        except Exception as e:  # a file that cannot even be scored is a FAIL
            print(f"\nFAIL  {p.name}\n    - could not evaluate: {e}")
            continue
        reports.append(rep)
        print_report(rep)

    if reports:
        print(f"\n{'='*74}")
        print("SUMMARY  (ranked by how hard the function punishes an off-track car)")
        print(f"{'='*74}")
        print(f"  {'result':<6} {'function':<48} {'offtrack':>9} {'reversed':>9}")
        ranked = sorted(
            reports,
            key=lambda r: (not r.passed, r.margins.get("offtrack", 9), r.margins.get("reversed", 9)),
        )
        for r in ranked:
            off = r.margins.get("offtrack")
            rev = r.margins.get("reversed")
            print(f"  {'PASS' if r.passed else 'FAIL':<6} {r.function:<48} "
                  f"{(f'{off*100:.0f} %' if off is not None else '-'):>9} "
                  f"{(f'{rev*100:.0f} %' if rev is not None else '-'):>9}")
        n_pass = sum(1 for r in reports if r.passed)
        print(f"\n  {n_pass} of {len(reports)} functions passed every gate.")

    if args.json:
        Path(args.json).write_text(
            json.dumps([asdict(r) for r in reports], indent=2), encoding="utf-8")
        print(f"\n  report written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
