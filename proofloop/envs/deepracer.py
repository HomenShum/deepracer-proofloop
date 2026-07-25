"""
Environment 1: DeepRacer reward-function design.

The artifact is a reward function. The verifiable reward is the lap time of a
policy trained to maximise that reward function. No model judges it.

This file adapts the existing evaluator to the ProofLoop interface. The loop in
proofloop/core.py does not know anything about racing. It drives this
environment and the travelling-salesman environment with the same code.

Baseline: 18.47 s, the human reward function `opt9`, trained.
Bound:    18.147 s, the sum of the racing line's own per-point times.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "proofloop"))
sys.path.insert(0, str(ROOT / "search"))

from loop import Gate, Result                                   # noqa: E402
from evaluator import evaluate, HUMAN_BEST, PHYSICAL_FLOOR      # noqa: E402

# Map an evaluator failure string onto a mechanism CLASS. The memory groups by
# class, so the proposer sees "this kind of mistake failed four times" instead
# of four unrelated rounds.
MECHANISMS = [
    ("G0 rejected by the allowlist", "disallowed_code"),
    ("G1 does not import", "syntax_error"),
    ("G3 pays per step", "step_farming"),
    ("does not finish a lap", "early_termination"),
    ("training failed", "runtime_error"),
]


def _mechanism(failures: list[str]) -> str:
    for f in failures:
        for needle, name in MECHANISMS:
            if needle in f:
                return name
    return "unknown"


class DeepRacerEnv:
    """Reward-function design, scored by a trained lap time."""

    def __init__(self, inner_iters: int = 6, inner_pop: int = 24):
        self.inner_iters = inner_iters
        self.inner_pop = inner_pop

    def name(self) -> str:
        return "deepracer"

    def baseline(self) -> float:
        return HUMAN_BEST

    def bound(self) -> float | None:
        # None means "no known bound", and rule R5 cannot run.
        #
        # 18.147 s was used here as a physical floor. It is not one. It is the
        # time to follow the racing line at its own speeds, and the corridor
        # permits cutting inside that line: a trained policy drove 40.614 m
        # against the line's 41.509 m. A legitimate 17.5 s lap would have been
        # rejected as an "evaluator violation" by a bound that was simply wrong.
        #
        # A wrong bound is worse than no bound. It rejects true results silently.
        return None

    def propose_context(self) -> str:
        return (
            "Write a Python function `reward_function(params)` for AWS DeepRacer.\n"
            "A policy is trained to MAXIMISE THE CUMULATIVE SUM of your reward over an\n"
            "episode. The score is the lap time of that trained policy. Lower is better.\n"
            f"The human baseline is {HUMAN_BEST:.2f} s. The physical floor is "
            f"{PHYSICAL_FLOOR:.3f} s.\n\n"
            "Two failure modes are rejected automatically:\n"
            "  step farming     a net positive per-step reward makes a slower lap score\n"
            "                   higher, so the car learns to crawl.\n"
            "  early termination a net per-step reward that is too negative makes\n"
            "                   crashing on step one beat finishing. A finish bonus must\n"
            "                   exceed the whole accumulated penalty of a lap, about 270\n"
            "                   to 290 steps.\n\n"
            "params keys: all_wheels_on_track, x, y, closest_waypoints,\n"
            "distance_from_center, is_offtrack, is_reversed, heading, progress, speed,\n"
            "steering_angle, steps, track_length, track_width, waypoints.\n"
            "Return 1e-3 at once if is_offtrack or not all_wheels_on_track.\n"
            "Import only `math`. Output one ```python block."
        )

    def score(self, artifact: str, seed: int = 0) -> Result:
        r = evaluate(artifact, iters=self.inner_iters, pop=self.inner_pop, seed=seed)

        failures = r["failures"]
        gates = [
            Gate("G0_allowlist", not any("G0" in f for f in failures),
                 next((f for f in failures if "G0" in f), "allowlist clear")),
            Gate("G1_imports", not any("G1" in f for f in failures),
                 next((f for f in failures if "G1" in f), "imports and runs")),
            Gate("G3_not_step_farmer", not any("G3" in f for f in failures),
                 next((f for f in failures if "G3" in f),
                      f"rho(speed)={r['rho_speed']:+.3f} rho(steps)={r['rho_steps']:+.3f}"
                      if r["rho_speed"] is not None else "not measured")),
            Gate("G2_finishes", r["lap_time"] < float("inf"),
                 next((f for f in failures if "G2" in f),
                      f"trained lap {r['lap_time']:.2f}s"
                      if r["lap_time"] < float("inf") else "did not finish")),
        ]

        if not r["passed"]:
            return Result(float("inf"), gates, mechanism=_mechanism(failures))
        return Result(r["lap_time"], gates, mechanism=None)
