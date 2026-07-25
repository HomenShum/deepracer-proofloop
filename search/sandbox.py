"""
The tool allowlist, and process isolation for candidate code.

This module fixes two defects recorded in PLAN.md.

DEFECT 1. The allowlist was only a request.
    The prompt told the model to import only `math`. Nothing enforced the rule.
    A candidate could read files or open a network connection. NodeAgent
    enforces the allowlist BEFORE the model sees the tools. This module does
    the same, with the abstract syntax tree, before any code runs.

DEFECT 2. Candidate code ran in the main process.
    The loader imported model output directly. One bad candidate could stop the
    whole experiment, and a time limit inside the same process cannot stop an
    endless loop. Each candidate now runs in a subprocess with a real time
    limit.

This is the NodeAgent pattern written in plain Python. This repository imports
no Node package. A reader can copy the pattern into any project.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import textwrap
import uuid
from pathlib import Path

ALLOWED_IMPORTS = {"math"}

BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals",
    "input", "breakpoint", "memoryview", "vars",
}

DEFAULT_TIMEOUT_S = 20


class Rejected(Exception):
    """The candidate broke a rule before it ran."""


# ---------------------------------------------------------------------------
# Step 2. The allowlist check
# ---------------------------------------------------------------------------

def check_source(source: str) -> list[str]:
    """Return a list of reasons to reject. An empty list means the code passes.

    The check runs before the code runs. This is the point of the check.
    """
    reasons: list[str] = []

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"the source does not parse: {e.msg} on line {e.lineno}"]

    has_entry = any(
        isinstance(n, ast.FunctionDef) and n.name == "reward_function"
        for n in ast.walk(tree)
    )
    if not has_entry:
        reasons.append("the source does not define a function named reward_function")

    for node in ast.walk(tree):
        # An import of any module other than math.
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    reasons.append(f"it imports '{a.name}'. Only {sorted(ALLOWED_IMPORTS)} is allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                reasons.append(f"it imports from '{node.module}'. Only {sorted(ALLOWED_IMPORTS)} is allowed")

        # A call to a banned builtin.
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            reasons.append(f"it uses the banned name '{node.id}'")

        # An attribute that starts with two underscore characters.
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            reasons.append(f"it reads the private attribute '{node.attr}'")

    # Remove duplicates but keep the order.
    seen, unique = set(), []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Step 3. Process isolation
# ---------------------------------------------------------------------------

_RUNNER = textwrap.dedent('''
    import json, sys, math

    def main():
        payload = json.loads(sys.stdin.read())
        src, cases = payload["source"], payload["cases"]
        ns = {"math": math, "__name__": "candidate"}
        try:
            exec(compile(src, "<candidate>", "exec"), ns)
        except Exception as e:
            print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
            return
        fn = ns.get("reward_function")
        if fn is None:
            print(json.dumps({"ok": False, "error": "no reward_function after exec"}))
            return
        out, errors = [], 0
        for p in cases:
            try:
                out.append(float(fn(dict(p))))
            except Exception:
                out.append(0.0)
                errors += 1
        print(json.dumps({"ok": True, "rewards": out, "errors": errors}))

    main()
''')


def run_isolated(source: str, cases: list[dict], timeout_s: int = DEFAULT_TIMEOUT_S) -> dict:
    """Score a list of step dictionaries in a separate process.

    Returns {"ok": bool, "rewards": [...], "errors": int} or
            {"ok": False, "error": "..."}.

    A crash, a timeout, or unreadable output is a failure. It is never
    silently treated as a pass.
    """
    tmp = Path(tempfile.gettempdir()) / f"pdd_runner_{uuid.uuid4().hex[:8]}.py"
    tmp.write_text(_RUNNER, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", str(tmp)],
            input=json.dumps({"source": source, "cases": cases}),
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"the candidate did not stop within {timeout_s} seconds"}
    finally:
        tmp.unlink(missing_ok=True)

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        return {"ok": False, "error": f"the process failed: {tail[-1] if tail else proc.returncode}"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "error": "the process produced no readable result"}


def admit(source: str, timeout_s: int = DEFAULT_TIMEOUT_S) -> dict:
    """Run both checks. This is the gate that a candidate must pass first.

    Returns {"admitted": bool, "reasons": [...], "sha256": "..."}.
    """
    reasons = check_source(source)
    if reasons:
        return {"admitted": False, "reasons": reasons, "sha256": source_sha256(source)}

    # A single harmless call proves the code runs at all, in isolation.
    probe = [{
        "all_wheels_on_track": True, "x": 0.0, "y": 0.0,
        "closest_waypoints": [0, 1], "distance_from_center": 0.0,
        "is_crashed": False, "is_left_of_center": True, "is_offtrack": False,
        "is_reversed": False, "heading": 0.0, "progress": 1.0, "speed": 2.0,
        "steering_angle": 0.0, "steps": 1, "track_length": 10.0,
        "track_width": 1.2, "waypoints": [[0.0, 0.0], [1.0, 0.0]],
    }]
    res = run_isolated(source, probe, timeout_s=timeout_s)
    if not res.get("ok"):
        return {"admitted": False, "reasons": [res.get("error", "unknown failure")],
                "sha256": source_sha256(source)}
    return {"admitted": True, "reasons": [], "sha256": source_sha256(source)}
