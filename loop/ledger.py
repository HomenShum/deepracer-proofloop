"""
Append-only ledger for the ProofLoop.

The ledger enforces one rule that makes the whole loop meaningful:

    A prediction must be recorded BEFORE the result exists.

Without that ordering, "the agent improved the score" is unfalsifiable. The
agent can always explain a result after seeing it. Prediction error is the
only number that shows whether it understood the system or got lucky.

The file is JSONL. One line per event. Nothing is ever edited or deleted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class LedgerError(RuntimeError):
    """Raised when a round tries to break the ordering rule."""


def _digest(path: Path) -> str:
    """Content hash of the file under test, so a swap cannot go unnoticed."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


@dataclass
class Round:
    index: int
    hypothesis: str
    change: str
    predicted_mean: float
    target: str
    target_digest: str


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    # -- reading -------------------------------------------------------------

    def events(self) -> list[dict]:
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def _append(self, event: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, sort_keys=True) + "\n")

    def next_index(self) -> int:
        preds = [e for e in self.events() if e["type"] == "prediction"]
        return len(preds) + 1

    def open_prediction(self) -> dict | None:
        """The most recent prediction that has no result yet."""
        pending = None
        for e in self.events():
            if e["type"] == "prediction":
                pending = e
            elif e["type"] == "result" and pending and e["index"] == pending["index"]:
                pending = None
        return pending

    # -- writing -------------------------------------------------------------

    def predict(self, hypothesis: str, change: str, predicted_mean: float,
                target: Path) -> Round:
        """Record the hypothesis and the predicted score. Must come first."""
        if self.open_prediction() is not None:
            raise LedgerError(
                "a prediction is already open. Record its result before predicting again.")
        target = Path(target)
        rnd = Round(
            index=self.next_index(),
            hypothesis=hypothesis.strip(),
            change=change.strip(),
            predicted_mean=float(predicted_mean),
            target=target.name,
            target_digest=_digest(target),
        )
        self._append({"type": "prediction", **rnd.__dict__})
        return rnd

    def record(self, actual_mean: float, passed: bool, failures: list[str],
               target: Path) -> dict:
        """Record the measured result against the open prediction."""
        pending = self.open_prediction()
        if pending is None:
            raise LedgerError(
                "no open prediction. The prediction must be written before the run.")

        target = Path(target)
        digest_now = _digest(target)
        swapped = digest_now != pending["target_digest"]

        predicted = pending["predicted_mean"]
        error = actual_mean - predicted
        rel = abs(error) / abs(predicted) if predicted else float("inf")

        event = {
            "type": "result",
            "index": pending["index"],
            "predicted_mean": predicted,
            "actual_mean": float(actual_mean),
            "error": error,
            "relative_error": rel,
            "passed_gates": bool(passed),
            "failures": list(failures),
            "target": target.name,
            "target_digest": digest_now,
            "file_changed_after_prediction": swapped,
        }
        self._append(event)
        return event

    def judge(self, index: int, verdict: str, reasoning: str) -> None:
        """Record the fresh-context judge's decision. Default FAIL."""
        if verdict not in {"SUPPORTED", "NOT_SUPPORTED"}:
            raise LedgerError("verdict must be SUPPORTED or NOT_SUPPORTED")
        self._append({
            "type": "judgement",
            "index": index,
            "verdict": verdict,
            "reasoning": reasoning.strip(),
        })

    # -- reporting -----------------------------------------------------------

    def summary(self) -> dict:
        ev = self.events()
        results = [e for e in ev if e["type"] == "result"]
        judged = {e["index"]: e for e in ev if e["type"] == "judgement"}

        accepted, rejected, unjudged = 0, 0, 0
        for r in results:
            j = judged.get(r["index"])
            if j is None:
                unjudged += 1          # default FAIL: unjudged is not accepted
            elif j["verdict"] == "SUPPORTED" and r["passed_gates"]:
                accepted += 1
            else:
                rejected += 1

        errs = [abs(r["error"]) for r in results]
        rels = [r["relative_error"] for r in results if r["relative_error"] != float("inf")]
        return {
            "rounds": len(results),
            "accepted": accepted,
            "rejected": rejected,
            "unjudged": unjudged,
            "mean_abs_prediction_error": sum(errs) / len(errs) if errs else None,
            "mean_relative_error": sum(rels) / len(rels) if rels else None,
            "tampered_rounds": sum(1 for r in results if r["file_changed_after_prediction"]),
        }
