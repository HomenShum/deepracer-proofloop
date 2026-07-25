"""
ProofLoop core: the portable interface and the loop.

Read SPEC.md first. This file is the whole contract. It has no dependency
beyond the Python standard library, so it can be copied into another project
without bringing anything with it.

The loop improves an artifact against a VERIFIABLE reward and refuses to
accept an improvement it cannot prove. Every rule below exists because this
repository already made the matching mistake. CHANGELOG.md holds the evidence.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Protocol


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class Gate:
    name: str
    passed: bool
    reason: str


@dataclass
class Result:
    """One scored run of one artifact on one seed."""
    value: float                      # the verifiable reward. Lower is better.
    gates: list[Gate] = field(default_factory=list)
    mechanism: str | None = None      # the failure class, or None when it passed

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates) and self.value < float("inf")


class Environment(Protocol):
    """Five methods. Nothing else. See SPEC.md section 4."""

    def name(self) -> str: ...
    def baseline(self) -> float: ...
    def bound(self) -> float | None: ...
    def propose_context(self) -> str: ...
    def score(self, artifact: str, seed: int) -> Result: ...


# A proposer takes the context and the failure memory, and returns an artifact.
Proposer = Callable[[str, str], str]


# ---------------------------------------------------------------------------
# Memory. Keyed by MECHANISM, not by round.
# ---------------------------------------------------------------------------

class Memory:
    """Failure memory that survives a restart.

    Grouped by mechanism so the proposer sees "this class of mistake failed
    four times" instead of four separate rounds that look unrelated.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def record(self, mechanism: str, detail: str) -> None:
        e = self._data.setdefault(mechanism, {"count": 0, "examples": []})
        e["count"] += 1
        if detail not in e["examples"]:
            e["examples"] = (e["examples"] + [detail])[-3:]
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def brief(self) -> str:
        if not self._data:
            return "Nothing has failed yet."
        lines = ["These failure classes are already known. Do not repeat them:"]
        for m, e in sorted(self._data.items(), key=lambda kv: -kv[1]["count"]):
            lines.append(f"  - {m} (failed {e['count']} times): {e['examples'][-1][:110]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------

def write_receipt(dirpath: Path, payload: dict) -> Path:
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    sha = payload.get("artifact_sha256", "nosha")[:12]
    p = dirpath / f"receipt_{payload['env']}_{payload['round']:03d}_{sha}.json"
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

@dataclass
class LoopReport:
    env: str
    baseline: float
    bound: float | None
    rounds: int
    accepted: int
    best_value: float | None
    best_seed_values: list[float]
    best_artifact: str | None
    proven: bool
    proof_reasons: list[str]
    elapsed_s: float


def run_loop(env: Environment, proposer: Proposer, rounds: int = 8,
             seeds: int = 5, memory_path: Path | None = None,
             receipts_dir: Path | None = None, verbose: bool = True) -> LoopReport:
    """Improve an artifact against a verifiable reward, and prove it or refuse.

    R6: every candidate is scored on `seeds` seeds and judged on the MEAN.
    A best-of-N score is never reported as the outcome.
    """
    mem = Memory(memory_path or Path(f"memory/{env.name()}.json"))
    receipts = Path(receipts_dir or "receipts")
    t0 = time.time()

    best_value, best_seed_values, best_artifact = None, [], None
    accepted = 0

    for rnd in range(1, rounds + 1):
        artifact = proposer(env.propose_context(), mem.brief())
        sha = hashlib.sha256(artifact.encode("utf-8")).hexdigest()

        per_seed, gates, mechanism = [], [], None
        for s in range(seeds):
            r = env.score(artifact, seed=s)
            gates = r.gates                    # gates are seed independent
            if not r.passed:
                mechanism = r.mechanism or "unknown"
                break
            per_seed.append(r.value)

        if mechanism or len(per_seed) < seeds:
            mem.record(mechanism or "unknown",
                       next((g.reason for g in gates if not g.passed), "no reason given"))
            write_receipt(receipts, {
                "env": env.name(), "round": rnd, "artifact_sha256": sha,
                "accepted": False, "mechanism": mechanism,
                "gates": [asdict(g) for g in gates],
                "seed_values": per_seed, "baseline": env.baseline(),
                "bound": env.bound(), "created_at": time.time(),
            })
            if verbose:
                print(f"  round {rnd:>2}: REJECTED  [{mechanism}]")
            continue

        mean = statistics.fmean(per_seed)
        sd = statistics.pstdev(per_seed) if len(per_seed) > 1 else 0.0
        better = mean < env.baseline()
        if better and (best_value is None or mean < best_value):
            best_value, best_seed_values, best_artifact = mean, list(per_seed), artifact
        if better:
            accepted += 1

        write_receipt(receipts, {
            "env": env.name(), "round": rnd, "artifact_sha256": sha,
            "accepted": bool(better), "mechanism": None,
            "gates": [asdict(g) for g in gates],
            "seed_values": per_seed, "mean": mean, "stdev": sd,
            "baseline": env.baseline(), "bound": env.bound(),
            "created_at": time.time(),
        })
        if verbose:
            mark = "ACCEPTED" if better else "no gain"
            print(f"  round {rnd:>2}: {mark:>8}  mean {mean:.3f}  sd {sd:.3f}  "
                  f"(baseline {env.baseline():.3f})")

    # ---- the proof check. See SPEC.md section 7. ----
    reasons, proven = [], False
    if best_value is None:
        reasons.append("no candidate beat the baseline on the mean")
    else:
        if len(best_seed_values) < 5:
            reasons.append(f"only {len(best_seed_values)} seeds; five is the minimum")
        if best_value >= env.baseline():
            reasons.append("the mean does not beat the baseline")
        b = env.bound()
        if b is not None and best_value < b:
            reasons.append(
                f"the result {best_value:.3f} is better than the known bound {b:.3f}. "
                "The evaluator is wrong, not the artifact")
        reasons.append("no independent party has reviewed this result")
        proven = len(reasons) == 1 and reasons[0].startswith("no independent")

    return LoopReport(
        env=env.name(), baseline=env.baseline(), bound=env.bound(), rounds=rounds,
        accepted=accepted, best_value=best_value, best_seed_values=best_seed_values,
        best_artifact=best_artifact, proven=proven, proof_reasons=reasons,
        elapsed_s=time.time() - t0,
    )


def print_report(rep: LoopReport) -> None:
    print(f"\n{'='*70}\n{rep.env}\n{'='*70}")
    print(f"  baseline        {rep.baseline:.3f}")
    print(f"  known bound     {rep.bound if rep.bound is None else f'{rep.bound:.3f}'}")
    print(f"  rounds          {rep.rounds}")
    print(f"  accepted        {rep.accepted}")
    if rep.best_value is not None:
        sd = statistics.pstdev(rep.best_seed_values) if len(rep.best_seed_values) > 1 else 0.0
        print(f"  best mean       {rep.best_value:.3f}  sd {sd:.3f}  "
              f"over {len(rep.best_seed_values)} seeds")
    print(f"  elapsed         {rep.elapsed_s:.1f}s")
    print(f"\n  PROVEN: {rep.proven}")
    for r in rep.proof_reasons:
        print(f"    - {r}")
