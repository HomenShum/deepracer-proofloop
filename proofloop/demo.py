"""
Run the ProofLoop against a real environment with a real proposer.

    python proofloop/demo.py --env tsp --rounds 6 --model z-ai/glm-5.2
    python proofloop/demo.py --env tsp --rounds 6 --proposer scripted

The scripted proposer needs no API key. It exists so a reader can run the loop
and see the gates work before they spend anything.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "envs"))
sys.path.insert(0, str(ROOT / "search"))

from loop import run_loop, print_report  # noqa: E402

# A ladder of scripted candidates: two that must be REJECTED, then real ones.
SCRIPTED = [
    # rejected by the allowlist
    "import os\ndef solve(cities):\n    return list(range(len(cities)))\n",
    # runs, but returns an invalid tour
    "def solve(cities):\n    return [0, 0, 1]\n",
    # nearest neighbour: equals the baseline, so no gain
    """import math
def solve(cities):
    un, order = set(range(1, len(cities))), [0]
    while un:
        last = order[-1]
        n = min(un, key=lambda c: math.dist(cities[last], cities[c]))
        order.append(n); un.remove(n)
    return order
""",
    # nearest neighbour plus 2-opt: should beat the baseline
    """import math
def solve(cities):
    def L(o):
        return sum(math.dist(cities[o[i]], cities[o[(i+1) % len(o)]]) for i in range(len(o)))
    un, order = set(range(1, len(cities))), [0]
    while un:
        last = order[-1]
        n = min(un, key=lambda c: math.dist(cities[last], cities[c]))
        order.append(n); un.remove(n)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(order) - 1):
            for j in range(i + 1, len(order)):
                cand = order[:i] + order[i:j+1][::-1] + order[j+1:]
                if L(cand) < L(order) - 1e-12:
                    order, improved = cand, True
    return order
""",
    # a cheat: hard-code a tour shorter than reality by lying about the output
    "def solve(cities):\n    return [0, 6, 2, 5, 3, 7, 8, 1, 4]\n",
]


def scripted_proposer():
    state = {"i": 0}

    def propose(context: str, memory: str) -> str:
        s = SCRIPTED[min(state["i"], len(SCRIPTED) - 1)]
        state["i"] += 1
        return s
    return propose


def llm_proposer(model: str):
    from llm_search import call_model, extract_code  # reuse the OpenRouter client

    def propose(context: str, memory: str) -> str:
        prompt = f"{context}\n\n{memory}\n\nWrite one function. Output one ```python block."
        raw = call_model(model, prompt)
        code = extract_code(raw)
        return code or "def solve(cities):\n    return list(range(len(cities)))\n"
    return propose


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="tsp", choices=["tsp", "deepracer"])
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--proposer", default="scripted", choices=["scripted", "llm"])
    ap.add_argument("--model", default="z-ai/glm-5.2")
    args = ap.parse_args()

    if args.env == "tsp":
        from tsp import TspEnv
        env = TspEnv()
    else:
        from deepracer import DeepRacerEnv
        env = DeepRacerEnv()

    print(f"\nProofLoop demo: {env.name()}")
    print(f"  baseline  {env.baseline():.4f}")
    print(f"  bound     {env.bound():.4f}")
    print(f"  proposer: {args.proposer}\n")

    proposer = (scripted_proposer() if args.proposer == "scripted"
                else llm_proposer(args.model))

    rep = run_loop(env, proposer, rounds=args.rounds, seeds=args.seeds,
                   memory_path=ROOT / "memory" / f"{env.name()}.json",
                   receipts_dir=ROOT / "receipts")
    print_report(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
