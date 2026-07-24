"""
Search A: automated search over reward-function DESIGNS.

Outer loop: CEM over reward-design weights. Inner loop: CEM over policy
parameters, which is what actually trains the car. The outer objective is the
trained lap time, never the reward value.

This is the honest version of "an agent improves the reward function". No LLM,
no API cost, fully autonomous, and it produces real reward function files that
sit next to the twelve human ones.

    python search/param_search.py --iters 8 --pop 12
    python search/param_search.py --iters 12 --pop 20 --inner-iters 8
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "search"))

from reward_template import GENOME, INIT_MU, INIT_SIGMA, clip, render  # noqa: E402
from evaluator import evaluate, HUMAN_BEST, PHYSICAL_FLOOR            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--inner-iters", type=int, default=6)
    ap.add_argument("--inner-pop", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="search/results_param.json")
    args = ap.parse_args()

    random.seed(args.seed)
    mu, sigma = list(INIT_MU), list(INIT_SIGMA)
    n_elite = max(3, args.pop // 4)

    print(f"\nSearch A: parametric reward-design search")
    print(f"  outer CEM {args.iters} x {args.pop}, inner CEM {args.inner_iters} x {args.inner_pop}")
    print(f"  human best  {HUMAN_BEST:.2f}s     physical floor {PHYSICAL_FLOOR:.3f}s")
    print(f"  objective: trained lap time. Lower is better.\n")

    best = {"lap_time": float("inf")}
    evaluated = 0
    first_match = None
    t0 = time.time()
    history = []

    for it in range(args.iters):
        cands = [clip([random.gauss(mu[i], sigma[i]) for i in range(len(mu))])
                 for _ in range(args.pop)]
        scored = []
        for g in cands:
            res = evaluate(render(g), iters=args.inner_iters, pop=args.inner_pop,
                           seed=args.seed)
            evaluated += 1
            res["genome"] = g
            scored.append(res)
            if res["passed"] and res["lap_time"] < best["lap_time"]:
                best = dict(res)
            if first_match is None and res["passed"] and res["lap_time"] <= HUMAN_BEST:
                first_match = {"evaluated": evaluated, "seconds": time.time() - t0,
                               "lap_time": res["lap_time"]}

        scored.sort(key=lambda r: r["lap_time"])
        elites = [r["genome"] for r in scored[:n_elite] if r["lap_time"] < float("inf")]
        if not elites:
            elites = [r["genome"] for r in scored[:n_elite]]
        for i in range(len(mu)):
            vals = [g[i] for g in elites]
            mu[i] = statistics.fmean(vals)
            sigma[i] = max(0.01, statistics.pstdev(vals) + 0.01)

        ok = [r for r in scored if r["passed"]]
        line = (f"  iter {it+1:>2}  best {scored[0]['lap_time']:.2f}s" if scored[0]["lap_time"] < float("inf")
                else f"  iter {it+1:>2}  best   DNF")
        print(f"{line}   passing {len(ok)}/{args.pop}   running best {best['lap_time']:.2f}s")
        history.append({"iter": it + 1, "passing": len(ok),
                        "best_this_iter": scored[0]["lap_time"] if scored[0]["lap_time"] < float("inf") else None,
                        "running_best": best["lap_time"] if best["lap_time"] < float("inf") else None})

    elapsed = time.time() - t0
    print(f"\n{'='*70}\nRESULT\n{'='*70}")
    print(f"  designs evaluated      {evaluated}")
    print(f"  wall clock             {elapsed:.1f}s")
    if best["lap_time"] < float("inf"):
        print(f"  best agent design      {best['lap_time']:.2f}s")
        print(f"  human best             {HUMAN_BEST:.2f}s")
        print(f"  physical floor         {PHYSICAL_FLOOR:.3f}s")
        delta = best["lap_time"] - HUMAN_BEST
        print(f"  vs human               {delta:+.2f}s "
              f"({'BEATS the human' if delta < 0 else 'does not beat the human'})")
        gap = (best["lap_time"] - PHYSICAL_FLOOR) / PHYSICAL_FLOOR * 100
        print(f"  off the physical floor {gap:+.1f}%")
        print(f"  weights                {dict(zip(GENOME, [round(v,4) for v in best['genome']]))}")
    else:
        print("  no design passed every gate")
    if first_match:
        print(f"\n  matched the human after {first_match['evaluated']} designs "
              f"and {first_match['seconds']:.0f}s")
    else:
        print(f"\n  never matched the human's {HUMAN_BEST:.2f}s")

    Path(args.out).write_text(json.dumps(
        {"best": {k: v for k, v in best.items() if k != "path"},
         "evaluated": evaluated, "elapsed_s": elapsed,
         "first_match": first_match, "history": history,
         "human_best": HUMAN_BEST, "physical_floor": PHYSICAL_FLOOR},
        indent=2, default=str), encoding="utf-8")
    print(f"\n  written to {args.out}")

    if best["lap_time"] < float("inf"):
        out = ROOT / "search" / "best_param_reward.py"
        out.write_text(render(best["genome"]), encoding="utf-8")
        print(f"  best design source -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
