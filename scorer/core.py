"""
Offline discrimination scorer for AWS DeepRacer reward functions.

You cannot compute a lap time from a reward function. A lap time needs training
and a simulator. So this module asks a different question, offline and for free:

    Does this reward function rank a good trajectory above a degraded one?

A reward function that scores an off-track car the same as a perfect lap is a
fake proof. It looks like it measures driving quality. It cannot fail correctly.

That is Rule 2 of Proof-Driven Development, applied to the reward function.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import math
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# DeepRacer physical defaults. Override per track when known.
DEFAULT_TRACK_WIDTH = 1.07
DEFAULT_MAX_SPEED = 4.0


# ----------------------------------------------------------------------------
# 1. Track extraction
# ----------------------------------------------------------------------------

def extract_racing_track(path: Path) -> list[list[float]]:
    """Pull the embedded racing_track literal out of a reward function file.

    Each row is [x, y, speed, time_from_previous_point].
    Every reward function carries its own line, so each one is scored against
    the track it was written for.
    """
    src = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"racing_track\s*=\s*(\[\s*\[.*?\]\s*\])", src, re.S)
    if not m:
        raise ValueError(f"no racing_track literal found in {path.name}")
    track = ast.literal_eval(m.group(1))
    if not track or len(track[0]) < 3:
        raise ValueError(f"racing_track in {path.name} has an unexpected shape")
    return [[float(v) for v in row] for row in track]


def track_length(track: list[list[float]]) -> float:
    total = 0.0
    for i in range(len(track)):
        x1, y1 = track[i][0], track[i][1]
        x2, y2 = track[(i + 1) % len(track)][0], track[(i + 1) % len(track)][1]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


# ----------------------------------------------------------------------------
# 2. Geometry helpers
# ----------------------------------------------------------------------------

def _heading_deg(p_from, p_to) -> float:
    return math.degrees(math.atan2(p_to[1] - p_from[1], p_to[0] - p_from[0]))


def _normal(p_prev, p_next) -> tuple[float, float]:
    """Unit vector perpendicular to the direction of travel (points left)."""
    dx, dy = p_next[0] - p_prev[0], p_next[1] - p_prev[1]
    n = math.hypot(dx, dy) or 1e-9
    return (-dy / n, dx / n)


def _closest_two(track, x, y) -> list[int]:
    d = [math.hypot(p[0] - x, p[1] - y) for p in track]
    i = min(range(len(d)), key=lambda k: d[k])
    prev_i, next_i = (i - 1) % len(track), (i + 1) % len(track)
    j = prev_i if d[prev_i] < d[next_i] else next_i
    return sorted([i, j]) if abs(i - j) == 1 else [i, j]


# ----------------------------------------------------------------------------
# 3. Trajectory synthesis
# ----------------------------------------------------------------------------

@dataclass
class Trajectory:
    name: str
    steps: list[dict]
    expect: str  # "best" | "degraded" | "invalid"
    note: str = ""


def _build(track, name, expect, note, *, lateral=0.0, speed_scale=1.0,
           oscillate=0.0, reverse=False, track_width=DEFAULT_TRACK_WIDTH,
           force_offtrack=False) -> Trajectory:
    n = len(track)
    order = list(range(n))[::-1] if reverse else list(range(n))
    steps: list[dict] = []

    for step_i, idx in enumerate(order):
        p = track[idx]
        p_prev = track[(idx - 1) % n]
        p_next = track[(idx + 1) % n]

        offset = lateral
        if oscillate:
            offset += oscillate * math.sin(step_i * 0.9)

        nx, ny = _normal(p_prev, p_next)
        x = p[0] + nx * offset
        y = p[1] + ny * offset

        heading = _heading_deg(p_prev, p_next)
        if reverse:
            heading = (heading + 180.0 + 180.0) % 360.0 - 180.0

        speed = max(0.1, float(p[2]) * speed_scale)
        dist_from_center = abs(offset)
        offtrack = force_offtrack or (dist_from_center > track_width / 2.0)

        steps.append({
            "all_wheels_on_track": not offtrack,
            "x": x,
            "y": y,
            "closest_waypoints": _closest_two(track, x, y),
            "distance_from_center": dist_from_center,
            "is_crashed": False,
            "is_left_of_center": offset > 0,
            "is_offtrack": offtrack,
            "is_reversed": reverse,
            "heading": heading,
            "progress": (step_i + 1) / len(order) * 100.0,
            "speed": speed,
            "steering_angle": 0.0 if not oscillate else 15.0 * math.cos(step_i * 0.9),
            "steps": step_i + 1,
            "track_length": track_length(track),
            "track_width": track_width,
            "waypoints": [[p[0], p[1]] for p in track],
        })
    return Trajectory(name=name, steps=steps, expect=expect, note=note)


def standard_suite(track, track_width=DEFAULT_TRACK_WIDTH) -> list[Trajectory]:
    """The trajectories every reward function must be able to tell apart."""
    return [
        _build(track, "optimal", "best",
               "follows the racing line at the prescribed speed",
               track_width=track_width),
        _build(track, "slow_75", "degraded",
               "correct line, 75 percent of the prescribed speed",
               speed_scale=0.75, track_width=track_width),
        _build(track, "slow_40", "degraded",
               "correct line, 40 percent of the prescribed speed",
               speed_scale=0.40, track_width=track_width),
        _build(track, "offset_15cm", "degraded",
               "0.15 m off the racing line, still on track",
               lateral=0.15, track_width=track_width),
        _build(track, "offset_40cm", "degraded",
               "0.40 m off the racing line, near the edge",
               lateral=0.40, track_width=track_width),
        _build(track, "oscillating", "degraded",
               "weaves across the racing line, wastes steps",
               oscillate=0.25, track_width=track_width),
        _build(track, "offtrack", "invalid",
               "all wheels off the track",
               lateral=0.90, force_offtrack=True, track_width=track_width),
        _build(track, "reversed", "invalid",
               "drives the track backwards",
               reverse=True, track_width=track_width),
    ]


# ----------------------------------------------------------------------------
# 4. Harness
# ----------------------------------------------------------------------------

def load_reward_function(path: Path):
    """Import a reward function file in isolation.

    A fresh module name each time, because several of these functions keep
    state between calls (for example first_racingpoint_index).
    """
    mod_name = f"_rf_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(mod_name, None)
    fn = getattr(mod, "reward_function", None)
    if fn is None:
        raise AttributeError(f"{path.name} defines no reward_function")
    return fn


@dataclass
class TrajectoryResult:
    name: str
    expect: str
    total: float
    mean: float
    errors: int
    note: str = ""


def run_trajectory(path: Path, traj: Trajectory) -> TrajectoryResult:
    """Reload the function for every trajectory so stale state cannot leak.

    These reward functions print debug output on every step. That is normal for
    DeepRacer, where the logs are the training record. Here it is noise, so
    stdout is captured and discarded.
    """
    total, errors = 0.0, 0
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        fn = load_reward_function(path)
        for params in traj.steps:
            try:
                r = fn(dict(params))
                total += float(r)
            except Exception:
                errors += 1
    n = len(traj.steps) or 1
    return TrajectoryResult(traj.name, traj.expect, total, total / n, errors, traj.note)


# ----------------------------------------------------------------------------
# 5. Scoring
# ----------------------------------------------------------------------------

@dataclass
class Report:
    function: str
    results: list[TrajectoryResult]
    track_points: int
    passed: bool = False
    failures: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    margins: dict = field(default_factory=dict)

    @property
    def optimal(self) -> TrajectoryResult | None:
        return next((r for r in self.results if r.name == "optimal"), None)


def evaluate(path: Path, track_width=DEFAULT_TRACK_WIDTH) -> Report:
    track = extract_racing_track(path)
    suite = standard_suite(track, track_width)
    results = [run_trajectory(path, t) for t in suite]
    rep = Report(function=path.name, results=results, track_points=len(track))

    opt = rep.optimal
    if opt is None:
        rep.failures.append("no optimal trajectory produced")
        return rep

    base = opt.mean

    # GATE 1. The optimal trajectory must score highest.
    for r in results:
        if r.name != "optimal" and r.mean >= base:
            rep.failures.append(
                f"cannot discriminate: '{r.name}' scores {r.mean:.3f} "
                f">= optimal {base:.3f}")

    # GATE 2. An off-track car must be heavily penalised.
    # Rule 2: if the behaviour breaks, the proof must turn red.
    off = next((r for r in results if r.name == "offtrack"), None)
    if off:
        ratio = off.mean / base if base else 1.0
        rep.margins["offtrack"] = ratio
        if ratio > 0.25:
            rep.failures.append(
                f"does not fail closed: an off-track car still scores "
                f"{ratio*100:.0f} percent of optimal")

    # GATE 2b. Reversed driving is normally caught by the simulator, which ends
    # the episode. No reward function in this corpus reads params['is_reversed'].
    # So a weak reversed score is an OBSERVATION, not a failure. Scoring a
    # backwards lap ABOVE a perfect lap is still a defect.
    rev = next((r for r in results if r.name == "reversed"), None)
    if rev:
        ratio = rev.mean / base if base else 1.0
        rep.margins["reversed"] = ratio
        if ratio > 1.0:
            rep.failures.append(
                f"rewards driving backwards: 'reversed' scores "
                f"{ratio*100:.0f} percent of optimal, above a perfect lap")
        elif ratio > 0.25:
            rep.observations.append(
                f"does not detect reverse direction on its own "
                f"({ratio*100:.0f} percent of optimal). It relies on the "
                f"simulator to end the episode. Reads params['is_reversed']: no.")

    # GATE 3. Degradation must be monotonic where it is ordered.
    ordered = [("slow_75", "slow_40"), ("offset_15cm", "offset_40cm")]
    for better, worse in ordered:
        b = next((r for r in results if r.name == better), None)
        w = next((r for r in results if r.name == worse), None)
        if b and w and w.mean > b.mean:
            rep.failures.append(
                f"not monotonic: '{worse}' ({w.mean:.3f}) scores above "
                f"'{better}' ({b.mean:.3f})")

    # GATE 4. The function must actually run.
    for r in results:
        if r.errors:
            rep.failures.append(f"raised on {r.errors} steps of '{r.name}'")

    for r in results:
        if r.name != "optimal":
            rep.margins.setdefault(r.name, r.mean / base if base else 1.0)

    rep.passed = not rep.failures  # default FAIL: only passes with zero failures
    return rep
