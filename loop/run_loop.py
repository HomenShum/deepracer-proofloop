"""
The ProofLoop round protocol, for DeepRacer reward functions.

Agent-agnostic on purpose. The loop does not call a model. You bring any agent
you like, or do it by hand. What the loop enforces is the ordering and the
gates, because that is the part that makes the result mean something.

    1.  predict   record the hypothesis and the predicted score BEFORE the run
    2.  (edit)    you or your agent changes the reward function
    3.  measure   run the scorer, compare against the prediction
    4.  judge     a fresh context with no write access rules on the evidence
    5.  status    show prediction error and accepted rounds

Usage
-----
    python loop/run_loop.py prompt   --target data/reward_functions/x.py
    python loop/run_loop.py predict  --target ... --hypothesis "..." \
                                     --change "..." --predict 17.2
    python loop/run_loop.py measure  --target ...
    python loop/run_loop.py judge    --index 1 --verdict SUPPORTED --why "..."
    python loop/run_loop.py status
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scorer"))
sys.path.insert(0, str(ROOT / "loop"))

from core import evaluate  # noqa: E402
from ledger import Ledger, LedgerError  # noqa: E402

LEDGER_PATH = ROOT / "loop" / "ledger.jsonl"


PROPOSE_PROMPT = """\
You are improving one AWS DeepRacer reward function.

TARGET FILE
    {target}

CURRENT MEASURED BEHAVIOUR
{table}

GATES THIS FUNCTION MUST PASS
    1. The optimal trajectory scores highest.
    2. An off-track car scores at most 25 percent of optimal.
    2b. A reversed lap never outscores a perfect lap.
    3. More degradation scores lower than less degradation.
    4. The function raises on no step.

WHAT TO DO
    Propose exactly ONE change. Not two. One.
    State it in this form, and nothing else:

    HYPOTHESIS: <what is wrong now, and why your change fixes it>
    CHANGE:     <the specific edit, precise enough to apply>
    PREDICT:    <the mean reward you expect on the optimal trajectory, a number>

RULES
    Your prediction is recorded before the run and cannot be revised.
    Prediction error is scored. A change that improves the number while the
    prediction was wrong counts as luck, not understanding.
    Do not touch the scorer. You have no write access to it.
"""

JUDGE_PROMPT = """\
You are judging whether evidence supports a claim. You did not build this and
you have no write access.

CLAIM
    {hypothesis}

CHANGE APPLIED
    {change}

PREDICTED mean reward on the optimal trajectory: {predicted}
ACTUAL    mean reward on the optimal trajectory: {actual}
Prediction error: {error} ({relative:.1%})

GATES: {gate_state}
{failures}

FILE CHANGED AFTER THE PREDICTION WAS RECORDED: {tampered}

ANSWER
    VERDICT: SUPPORTED or NOT_SUPPORTED
    WHY: one paragraph.

Default to NOT_SUPPORTED. Answer SUPPORTED only if every gate passed AND the
result is close enough to the prediction to show the change was understood
rather than stumbled into. A large prediction error is grounds for
NOT_SUPPORTED even when the score improved.
"""


def _table(rep) -> str:
    base = rep.optimal.mean if rep.optimal else 0.0
    rows = []
    for r in rep.results:
        pct = "" if r.name == "optimal" else f"{(r.mean/base*100 if base else 0):.0f}% of optimal"
        rows.append(f"    {r.name:<14} {r.mean:>12.4f}   {pct}")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="ProofLoop round protocol.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prompt", help="print the proposal prompt for your agent")
    p.add_argument("--target", required=True)

    p = sub.add_parser("predict", help="record hypothesis and prediction BEFORE the run")
    p.add_argument("--target", required=True)
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--change", required=True)
    p.add_argument("--predict", type=float, required=True)

    p = sub.add_parser("measure", help="run the scorer and record the result")
    p.add_argument("--target", required=True)

    p = sub.add_parser("judge", help="record the fresh-context verdict")
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--verdict", required=True, choices=["SUPPORTED", "NOT_SUPPORTED"])
    p.add_argument("--why", required=True)

    sub.add_parser("status", help="show prediction error and accepted rounds")

    args = ap.parse_args()
    led = Ledger(LEDGER_PATH)

    if args.cmd == "prompt":
        target = Path(args.target)
        rep = evaluate(target)
        print(PROPOSE_PROMPT.format(target=target, table=_table(rep)))
        return 0

    if args.cmd == "predict":
        try:
            rnd = led.predict(args.hypothesis, args.change, args.predict, Path(args.target))
        except LedgerError as e:
            print(f"REFUSED: {e}")
            return 1
        print(f"round {rnd.index} open. predicted mean {rnd.predicted_mean:.4f} "
              f"on {rnd.target} (digest {rnd.target_digest}).")
        print("Now apply the change, then run: measure")
        return 0

    if args.cmd == "measure":
        target = Path(args.target)
        rep = evaluate(target)
        actual = rep.optimal.mean if rep.optimal else 0.0
        try:
            ev = led.record(actual, rep.passed, rep.failures, target)
        except LedgerError as e:
            print(f"REFUSED: {e}")
            print("The prediction must exist before the result. That is the point.")
            return 1
        print(f"round {ev['index']}")
        print(f"  predicted {ev['predicted_mean']:.4f}")
        print(f"  actual    {ev['actual_mean']:.4f}")
        print(f"  error     {ev['error']:+.4f}  ({ev['relative_error']:.1%})")
        print(f"  gates     {'PASS' if ev['passed_gates'] else 'FAIL'}")
        for f in ev["failures"]:
            print(f"    - {f}")
        if ev["file_changed_after_prediction"]:
            print("  WARNING: the file changed after the prediction was recorded.")
        print(f"\nNow judge it with a fresh context:")
        print(JUDGE_PROMPT.format(
            hypothesis=next(e["hypothesis"] for e in led.events()
                            if e["type"] == "prediction" and e["index"] == ev["index"]),
            change=next(e["change"] for e in led.events()
                        if e["type"] == "prediction" and e["index"] == ev["index"]),
            predicted=f"{ev['predicted_mean']:.4f}",
            actual=f"{ev['actual_mean']:.4f}",
            error=f"{ev['error']:+.4f}",
            relative=ev["relative_error"],
            gate_state="PASS" if ev["passed_gates"] else "FAIL",
            failures="\n".join(f"    - {f}" for f in ev["failures"]) or "    (none)",
            tampered="yes" if ev["file_changed_after_prediction"] else "no",
        ))
        return 0

    if args.cmd == "judge":
        try:
            led.judge(args.index, args.verdict, args.why)
        except LedgerError as e:
            print(f"REFUSED: {e}")
            return 1
        print(f"round {args.index} judged {args.verdict}.")
        return 0

    if args.cmd == "status":
        s = led.summary()
        print("\nPROOFLOOP STATUS")
        print(f"  rounds run            {s['rounds']}")
        print(f"  accepted              {s['accepted']}")
        print(f"  rejected              {s['rejected']}")
        print(f"  unjudged (not passed) {s['unjudged']}")
        if s["mean_abs_prediction_error"] is not None:
            print(f"  mean |prediction error| {s['mean_abs_prediction_error']:.4f}")
            print(f"  mean relative error     {s['mean_relative_error']:.1%}")
        if s["tampered_rounds"]:
            print(f"  TAMPERED ROUNDS       {s['tampered_rounds']}")
        print()
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
