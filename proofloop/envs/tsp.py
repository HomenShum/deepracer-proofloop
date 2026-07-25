"""
Environment 2: a small travelling-salesman instance.

This exists to test whether the interface is portable. It shares nothing with
DeepRacer. There is no simulator, no policy, and no training. The artifact is
a Python heuristic. The verifiable reward is the tour length.

It also tests rule R5, the known bound. The instance has nine cities, so the
optimal tour is computed by brute force. Any candidate that reports a shorter
tour is proof that the evaluator is wrong, not that the heuristic is clever.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "search"))

from loop import Gate, Result          # noqa: E402
from sandbox import admit, run_isolated  # noqa: E402

# A fixed nine-city instance. Fixed so the result is reproducible.
CITIES = [(0.0, 0.0), (2.0, 4.0), (5.0, 1.0), (6.0, 5.0), (1.0, 7.0),
          (8.0, 3.0), (3.0, 2.0), (7.0, 8.0), (4.0, 6.0)]


def tour_length(order) -> float:
    return sum(math.dist(CITIES[order[i]], CITIES[order[(i + 1) % len(order)]])
               for i in range(len(order)))


def _optimal() -> float:
    best = float("inf")
    for p in itertools.permutations(range(1, len(CITIES))):
        best = min(best, tour_length((0,) + p))
    return best


OPTIMAL = _optimal()
# The baseline is nearest neighbour from city 0, the obvious first heuristic.
def _nearest_neighbour() -> float:
    unvisited, order = set(range(1, len(CITIES))), [0]
    while unvisited:
        last = order[-1]
        nxt = min(unvisited, key=lambda c: math.dist(CITIES[last], CITIES[c]))
        order.append(nxt)
        unvisited.remove(nxt)
    return tour_length(order)


BASELINE = _nearest_neighbour()


class TspEnv:
    """The artifact is a function `solve(cities) -> list[int]`."""

    def name(self) -> str:
        return "tsp9"

    def baseline(self) -> float:
        return BASELINE

    def bound(self) -> float | None:
        return OPTIMAL

    def propose_context(self) -> str:
        return (
            f"Write a Python function `solve(cities)` that returns a tour as a list of "
            f"indices. `cities` is a list of {len(CITIES)} (x, y) tuples. The tour must "
            f"visit every city exactly once and return to the start.\n"
            f"The score is the tour length. Lower is better.\n"
            f"Nearest neighbour scores {BASELINE:.4f}. Beat it.\n"
            f"Import only `math` and `itertools`. Define `solve` and nothing that runs "
            f"at import time."
        )

    def score(self, artifact: str, seed: int = 0) -> Result:
        gates: list[Gate] = []

        verdict = admit_tsp(artifact)
        gates.append(Gate("G0_allowlist", verdict["ok"], verdict["reason"]))
        if not verdict["ok"]:
            return Result(float("inf"), gates, mechanism="disallowed_code")

        runner = (
            artifact
            + "\n\nimport json,sys\n"
            + f"CITIES={CITIES!r}\n"
            + "t=solve(CITIES)\n"
            + "print(json.dumps({'tour':list(t)}))\n"
        )
        res = run_isolated_text(runner, timeout_s=10)
        gates.append(Gate("G1_runs", res["ok"], res.get("reason", "ran")))
        if not res["ok"]:
            return Result(float("inf"), gates, mechanism=res.get("mechanism", "runtime_error"))

        tour = res["tour"]
        valid = sorted(tour) == list(range(len(CITIES)))
        gates.append(Gate("G2_valid_tour", valid,
                          "visits every city once" if valid else f"invalid tour: {tour}"))
        if not valid:
            return Result(float("inf"), gates, mechanism="invalid_output")

        length = tour_length(tour)
        # R5. A tour shorter than the brute-force optimum means the evaluator is wrong.
        sane = length >= OPTIMAL - 1e-9
        gates.append(Gate("G3_within_bound", sane,
                          f"length {length:.4f}, optimum {OPTIMAL:.4f}"))
        if not sane:
            return Result(float("inf"), gates, mechanism="evaluator_violation")

        return Result(length, gates, mechanism=None)


def admit_tsp(source: str) -> dict:
    """Same idea as the DeepRacer allowlist, with `itertools` also permitted."""
    import ast
    allowed = {"math", "itertools", "json", "sys"}
    banned = {"open", "exec", "eval", "compile", "__import__", "globals"}
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"ok": False, "reason": f"does not parse: {e.msg} line {e.lineno}"}
    if not any(isinstance(n, ast.FunctionDef) and n.name == "solve" for n in ast.walk(tree)):
        return {"ok": False, "reason": "no function named solve"}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] not in allowed:
                    return {"ok": False, "reason": f"imports '{a.name}'"}
        elif isinstance(n, ast.ImportFrom):
            if (n.module or "").split(".")[0] not in allowed:
                return {"ok": False, "reason": f"imports from '{n.module}'"}
        elif isinstance(n, ast.Name) and n.id in banned:
            return {"ok": False, "reason": f"uses banned name '{n.id}'"}
    return {"ok": True, "reason": "allowlist clear"}


def run_isolated_text(program: str, timeout_s: int = 10) -> dict:
    """Run a self-contained program in a subprocess and read one JSON line."""
    import json as _json
    import subprocess
    import sys as _sys
    import tempfile
    import uuid

    tmp = Path(tempfile.gettempdir()) / f"pl_tsp_{uuid.uuid4().hex[:8]}.py"
    tmp.write_text(program, encoding="utf-8")
    try:
        p = subprocess.run([_sys.executable, "-I", "-S", str(tmp)],
                           capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": f"did not stop within {timeout_s}s",
                "mechanism": "timeout"}
    finally:
        tmp.unlink(missing_ok=True)
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()
        return {"ok": False, "reason": tail[-1] if tail else "nonzero exit",
                "mechanism": "runtime_error"}
    try:
        d = _json.loads(p.stdout.strip().splitlines()[-1])
        return {"ok": True, "tour": d["tour"], "reason": "ran"}
    except Exception:
        return {"ok": False, "reason": "no readable output", "mechanism": "bad_output"}
