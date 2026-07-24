"""
The shared evaluator. Both the parametric search and the LLM search use this,
so their results are directly comparable.

A candidate reward function is scored by ONE number: the lap time of a policy
trained to maximise it. Not the reward. Not a correlation. The lap time.

Gates, all derived from a failure already observed in this repo:

  G1 imports and runs          a candidate that raises is worthless
  G2 the trained policy finishes  reward_function.py trains to a DNF
  G3 not a step farmer         rho(reward, steps) must not exceed
                               rho(reward, speed). Seven of twelve human
                               functions fail this.
  G4 beats the incumbent       on trained lap time, nothing else

Default FAIL. A candidate that does not clear every gate scores infinity.
"""

from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scorer"))
sys.path.insert(0, str(ROOT / "sim"))

from core import load_reward_function                    # noqa: E402
from track_sim import load_line, load_actions, spearman  # noqa: E402
import train as trainer                                  # noqa: E402

LINE = load_line(ROOT / "data" / "lines" / "optimals_newest_Ross_racing_line.txt")
ACTS = load_actions(ROOT / "data" / "lines" / "AS21_newest_Ross_racing_line.txt")
PHYSICAL_FLOOR = 18.147          # sum of the racing line's own per-point times
HUMAN_BEST = 18.47               # opt9, trained, under the grip limit

_SCRATCH = Path(tempfile.gettempdir()) / "pdd_candidates"
_SCRATCH.mkdir(parents=True, exist_ok=True)


def write_candidate(source: str, name: str | None = None) -> Path:
    p = _SCRATCH / f"cand_{name or uuid.uuid4().hex[:10]}.py"
    p.write_text(source, encoding="utf-8")
    return p


def _step_farmer(path: Path) -> tuple[bool, float, float]:
    """Does this reward pay for a longer lap rather than a faster one?

    Scores a fixed spread of completing laps and compares the correlation with
    step count against the correlation with speed.
    """
    fn = load_reward_function(path)
    finishers = []
    for la in (4, 8, 12):
        for sp in (0.45, 0.65, 0.85):
            for caution in (0.4, 1.4):
                fin, steps, lap, tr = trainer.rollout(LINE, ACTS, [float(la), sp, 0.0, caution, 1.0])
                if fin:
                    finishers.append((steps, lap, tr))
    if len(finishers) < 4:
        return False, 0.0, 0.0        # cannot tell; do not fail on this alone

    import contextlib, io
    totals = []
    with contextlib.redirect_stdout(io.StringIO()):
        for _, _, tr in finishers:
            t = 0.0
            for p in tr:
                try:
                    t += float(fn(dict(p)))
                except Exception:
                    pass
            totals.append(t)
    steps = [float(f[0]) for f in finishers]
    speeds = [-f[1] for f in finishers]
    rho_steps = spearman(totals, steps)
    rho_speed = spearman(totals, speeds)
    return (rho_steps > rho_speed), rho_speed, rho_steps


def evaluate(source: str, iters=6, pop=24, seed=0, name=None) -> dict:
    """Score one candidate reward function. Lower lap_time is better."""
    out = {
        "lap_time": float("inf"), "passed": False, "failures": [],
        "rho_speed": None, "rho_steps": None, "path": None,
    }
    try:
        path = write_candidate(source, name)
        out["path"] = str(path)
    except Exception as e:
        out["failures"].append(f"could not write candidate: {e}")
        return out

    # G1 imports and runs
    try:
        load_reward_function(path)
    except Exception as e:
        out["failures"].append(f"G1 does not import: {type(e).__name__}: {e}")
        return out

    # G3 step farmer
    try:
        farms, rho_speed, rho_steps = _step_farmer(path)
        out["rho_speed"], out["rho_steps"] = rho_speed, rho_steps
        if farms:
            out["failures"].append(
                f"G3 pays per step: rho(steps)={rho_steps:+.3f} exceeds rho(speed)={rho_speed:+.3f}")
    except Exception as e:
        out["failures"].append(f"G3 could not be checked: {e}")

    # G2 the trained policy finishes
    try:
        r = trainer.train(path, iters=iters, pop=pop, seed=seed)
    except Exception as e:
        out["failures"].append(f"G2 training failed: {type(e).__name__}: {e}")
        return out

    if not r["lap_time"]:
        out["failures"].append("G2 the trained policy does not finish a lap")
        return out

    out["lap_time"] = r["lap_time"]
    out["genome_policy"] = r["final_genome"]
    out["passed"] = not out["failures"]
    return out
