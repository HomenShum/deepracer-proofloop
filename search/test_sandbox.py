"""
Adversarial test for the allowlist and the process isolation.

The plan says a gate must be able to fail. This file tries to break each gate
on purpose. If a hostile candidate passes, the gate is decoration.

    python search/test_sandbox.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sandbox import admit, check_source, run_isolated  # noqa: E402

HONEST = """
import math

def reward_function(params):
    if params.get("is_offtrack"):
        return 1e-3
    return 1.0 + params.get("speed", 0.0) / 4.0
"""

ATTACKS = [
    ("imports os", """
import os
def reward_function(params):
    return 1.0
"""),
    ("imports from subprocess", """
from subprocess import run
def reward_function(params):
    return 1.0
"""),
    ("calls open", """
def reward_function(params):
    f = open("/etc/passwd")
    return 1.0
"""),
    ("calls eval", """
def reward_function(params):
    return eval("1.0")
"""),
    ("reaches a private attribute", """
def reward_function(params):
    return float(len(params.__class__.__mro__))
"""),
    ("no reward_function", """
def score(params):
    return 1.0
"""),
    ("does not parse", """
def reward_function(params)
    return 1.0
"""),
]

ENDLESS = """
import math
def reward_function(params):
    while True:
        pass
"""


def main() -> int:
    failures = 0

    print("1. An honest candidate must be admitted.")
    r = admit(HONEST)
    ok = r["admitted"]
    print(f"   {'PASS' if ok else 'FAIL'}  admitted={r['admitted']} sha={r['sha256'][:12]}")
    if not ok:
        print(f"        reasons: {r['reasons']}")
        failures += 1

    print("\n2. Every hostile candidate must be rejected before it runs.")
    for label, src in ATTACKS:
        reasons = check_source(src)
        rejected = bool(reasons)
        print(f"   {'PASS' if rejected else 'FAIL'}  {label:32s} "
              f"{reasons[0][:56] if reasons else 'ADMITTED, which is wrong'}")
        if not rejected:
            failures += 1

    print("\n3. A candidate that never stops must time out, not hang the harness.")
    res = run_isolated(ENDLESS, [{"speed": 1.0}], timeout_s=5)
    timed_out = (not res.get("ok")) and "did not stop" in str(res.get("error", ""))
    print(f"   {'PASS' if timed_out else 'FAIL'}  {res.get('error', res)}")
    if not timed_out:
        failures += 1

    print("\n4. The harness must still work after the attacks.")
    r = admit(HONEST)
    print(f"   {'PASS' if r['admitted'] else 'FAIL'}  honest candidate still admitted")
    if not r["admitted"]:
        failures += 1

    print(f"\n{'ALL GATES HOLD' if failures == 0 else f'{failures} GATE FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
