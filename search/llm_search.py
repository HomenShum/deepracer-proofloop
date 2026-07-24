"""
Search B: an LLM writes reward functions, the same evaluator scores them.

Search A could only tune WEIGHTS inside a fixed template. It plateaued at
19.13s against the human's 18.47s, because the human's best design uses a
functional FORM the template cannot express: a finish bonus that scales with
how far under a target step count the lap came in.

An LLM writes arbitrary code, so it can invent forms. That is the only reason
this search is interesting. If it merely retunes weights it will land where A
landed.

Every candidate goes through the identical evaluator and the identical gates.
The objective is the trained lap time and nothing else.

NOTE ON SAFETY: generated code is imported and executed locally. It is model
output, so treat it as untrusted. The prompt forbids imports beyond math and
the loader runs it in-process; a hostile candidate could do more than compute
a reward. Acceptable on a local box for a bounded experiment, not acceptable
unattended or on anything shared.

    python search/llm_search.py --model z-ai/glm-5.2 --rounds 8
    python search/llm_search.py --model google/gemma-4-31b-it:free --rounds 8
    python search/llm_search.py --compare
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "search"))

from evaluator import evaluate, HUMAN_BEST, PHYSICAL_FLOOR  # noqa: E402

API_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM = """You write AWS DeepRacer reward functions. You output Python and nothing else.

Hard requirements:
- Define exactly: def reward_function(params)
- Import only `math`. No other imports. No file, network, or system access.
- Return a float.
- params keys available: all_wheels_on_track, x, y, closest_waypoints,
  distance_from_center, is_offtrack, is_reversed, heading, progress, speed,
  steering_angle, steps, track_length, track_width, waypoints
- Return 1e-3 immediately if is_offtrack or not all_wheels_on_track.

Output format: one ```python fenced block. No prose before or after."""

TASK = """Design a reward function that, when a policy is trained to MAXIMISE ITS
CUMULATIVE SUM over an episode, produces the FASTEST completed lap.

The physical floor on this track is {floor:.3f}s. The best human-written reward
function trains to {human:.2f}s. Beat {human:.2f}s.

The scoring loop trains a driving policy against your reward, then times the lap.
Your reward value is never the score. The lap time is the score.

Two failure modes are already known and both are automatically rejected:

1. STEP FARMING. If the net per-step reward is positive, a slower lap has more
   steps and therefore a larger sum, so training makes the car crawl. Seven of
   twelve human functions failed exactly this way.

2. EARLY TERMINATION. If the net per-step reward is too negative, crashing on
   step one stops the loss and beats finishing. A finish bonus must exceed the
   total accumulated penalty of a full lap (roughly 270 to 290 steps).

The valid region is narrow and sits between them.

Known: a purely parametric search over
  reward = w_const + w_dist*line_term + w_speed*speed_term + w_align*align_term
           (+ w_finish at progress 100)
converged to 19.13s and could not reach 18.47s. Tuning those weights is a dead
end. The human's winning design uses a different FORM: the finish bonus scales
with how far under a target step count the lap finished, which encodes time
directly instead of approximating it.

{memory}

Write one reward function. Make it different from what has already failed."""


def _key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not k:
        p = Path("/tmp/.orkey")
        if p.exists():
            k = p.read_text().strip()
    if not k:
        sys.exit("OPENROUTER_API_KEY not set and /tmp/.orkey missing")
    return k


def call_model(model: str, prompt: str, timeout=180) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": 8000,
    }).encode()
    req = urllib.request.Request(
        API_URL, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_key()}",
                 "HTTP-Referer": "https://github.com/HomenShum/deepracer-proofloop",
                 "X-Title": "DeepRacer ProofLoop"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    msg = d["choices"][0]["message"]
    # Reasoning models can spend the whole token budget on `reasoning` and
    # return empty `content`. Fall back to reasoning so the code can still be
    # recovered, and never return None.
    return (msg.get("content") or msg.get("reasoning") or "")


def extract_code(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    code = m.group(1) if m else text
    return code.strip() if "def reward_function" in code else None


def format_memory(hist) -> str:
    if not hist:
        return "Nothing has been tried yet."
    lines = ["Already tried, do not repeat these mistakes:"]
    for h in hist[-6:]:
        if h["lap_time"] < float("inf"):
            lines.append(f"  - {h['note']} -> trained lap {h['lap_time']:.2f}s")
        else:
            why = h["failures"][0] if h["failures"] else "failed"
            lines.append(f"  - {h['note']} -> REJECTED: {why}")
    return "\n".join(lines)


def run(model: str, rounds: int, inner_iters=6, inner_pop=24) -> dict:
    hist, best = [], {"lap_time": float("inf"), "model": model}
    t0 = time.time()
    print(f"\n=== {model} ===")
    for i in range(rounds):
        prompt = TASK.format(floor=PHYSICAL_FLOOR, human=HUMAN_BEST,
                             memory=format_memory(hist))
        try:
            raw = call_model(model, prompt)
        except urllib.error.HTTPError as e:
            print(f"  round {i+1}: HTTP {e.code} {e.read()[:120].decode(errors='replace')}")
            continue
        except Exception as e:
            print(f"  round {i+1}: {type(e).__name__}: {e}")
            continue

        code = extract_code(raw)
        if not code:
            print(f"  round {i+1}: no reward_function in the reply")
            hist.append({"note": "reply contained no function", "lap_time": float("inf"),
                         "failures": ["no code"]})
            continue

        res = evaluate(code, iters=inner_iters, pop=inner_pop,
                       name=f"{model.replace('/','_').replace(':','_')}_{i+1}")
        note = code.splitlines()[0][:70] if code.splitlines() else "candidate"
        hist.append({"note": f"round {i+1}", "lap_time": res["lap_time"],
                     "failures": res["failures"], "code": code})
        if res["passed"] and res["lap_time"] < best["lap_time"]:
            best = {**res, "model": model, "code": code, "round": i + 1}

        lap = f"{res['lap_time']:.2f}s" if res["lap_time"] < float("inf") else "FAIL"
        flag = "" if res["passed"] else f"   [{res['failures'][0][:58]}]" if res["failures"] else ""
        print(f"  round {i+1}: {lap:>8}{flag}")

    best["elapsed_s"] = time.time() - t0
    best["rounds"] = rounds
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="z-ai/glm-5.2")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--compare", action="store_true",
                    help="run GLM 5.2 plus a set of free models")
    ap.add_argument("--out", default="search/results_llm.json")
    args = ap.parse_args()

    models = ([args.model] if not args.compare else [
        "z-ai/glm-5.2",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "google/gemma-4-31b-it:free",
        "openai/gpt-oss-20b:free",
    ])

    print(f"Search B: LLM-written reward functions")
    print(f"  human best {HUMAN_BEST:.2f}s   physical floor {PHYSICAL_FLOOR:.3f}s")
    print(f"  parametric search A plateaued at 19.13s")

    results = []
    for m in models:
        results.append(run(m, args.rounds))

    print(f"\n{'='*72}\nRESULT\n{'='*72}")
    print(f"  {'model':<44} {'best lap':>10} {'vs human':>10}")
    for r in sorted(results, key=lambda r: r["lap_time"]):
        if r["lap_time"] < float("inf"):
            print(f"  {r['model']:<44} {r['lap_time']:>9.2f}s {r['lap_time']-HUMAN_BEST:>+9.2f}s")
        else:
            print(f"  {r['model']:<44} {'none passed':>10} {'-':>10}")

    ok = [r for r in results if r["lap_time"] < float("inf")]
    if ok:
        b = min(ok, key=lambda r: r["lap_time"])
        print(f"\n  best overall {b['lap_time']:.2f}s from {b['model']} (round {b.get('round')})")
        print(f"  human {HUMAN_BEST:.2f}s   -> {'BEATS the human' if b['lap_time'] < HUMAN_BEST else 'does NOT beat the human'}")
        (ROOT / "search" / "best_llm_reward.py").write_text(b["code"], encoding="utf-8")
        print(f"  source -> search/best_llm_reward.py")
    else:
        print("\n  no model produced a passing design")

    Path(args.out).write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "path"} for r in results],
        indent=2, default=str), encoding="utf-8")
    print(f"  written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
