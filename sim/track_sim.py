"""
A kinematic DeepRacer simulator, and the test that matters.

WHAT THIS IS FOR
    A reward function has exactly one job: rank fast trajectories above slow
    ones. If maximising the reward does not minimise lap time, the function is
    wrong, whatever else it does.

    That is measurable without AWS. Generate a family of driving policies,
    simulate each to get a real lap time, score each with the reward function,
    and measure the rank correlation between reward and speed.

    A perfect reward function scores Spearman rho = +1.0.

WHAT THIS IS NOT
    This is not the AWS simulator. Absolute lap times here will not match AWS
    numbers, and are not meant to. What transfers is the RANKING: whether a
    reward function prefers the trajectory that is actually faster.

CONSTANTS, and where they come from
    15 Hz timestep      from the reward functions themselves, which compute
                        current_actual_time = (step_count - 1) / 15
    21 discrete actions from the AS21 action-space files in data/lines
    speed 1.3 to 3.95   same source
    steering +/- 30 deg same source
    wheelbase 0.165 m   AWS DeepRacer is a 1/18 scale car
"""

from __future__ import annotations

import ast
import io
import json
import math
import contextlib
from dataclasses import dataclass, field
from pathlib import Path

DT = 1.0 / 15.0          # DeepRacer control rate
WHEELBASE = 0.165        # metres, 1/18 scale car
MAX_STEER_DEG = 30.0
SPEED_TAU = 0.25         # first-order lag on reaching target speed, seconds
TRACK_HALF_WIDTH = 0.60  # drivable corridor either side of the racing line
MAX_STEPS = 3000         # a lap that takes longer than this is a DNF

# Grip limit, derived from the racing line data itself, not assumed.
# For every point on optimals_newest_Ross_racing_line.txt, v^2 / R peaks at
# 2.01 m/s^2. The line's speed profile (1.30 to 4.00 m/s) exists because of
# that limit. A simulator that ignores it lets the car corner at impossible
# speeds, and any search run against it will exploit exactly that.
#
# Without this cap the trained laps came out at 15.27 s against a theoretical
# perfect lap of 18.147 s, which is physically impossible and invalidated
# every result measured before it was added.
MAX_LATERAL_ACCEL = 2.01  # m/s^2, about 0.21 g


def _yaw_rate_deg(speed: float, steer_deg: float) -> float:
    """Heading change per second, limited by grip.

    The bicycle model wants v / L * tan(steer). Grip allows at most
    a_max / v of lateral acceleration worth of turning. Demanding more
    produces understeer: the car goes wide instead of turning, which is
    what puts a car off the track when a corner is taken too fast.
    """
    want = speed / WHEELBASE * math.tan(math.radians(steer_deg))
    if speed < 1e-6:
        return 0.0
    limit = MAX_LATERAL_ACCEL / speed          # rad/s
    return math.degrees(max(-limit, min(limit, want)))


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

def load_line(path: Path) -> list[list[float]]:
    """Load an optimals_*.txt racing line: [[x, y, speed, dt], ...]."""
    return [list(map(float, r)) for r in ast.literal_eval(path.read_text().strip())]


def load_actions(path: Path) -> list[tuple[float, float]]:
    """Load an AS21_*.txt action space as (steering_deg, speed) pairs."""
    d = json.loads(path.read_text())
    return [(float(a["steering_angle"]), float(a["speed"])) for a in d]


def _closest_index(line, x, y, near=None, window=25) -> int:
    """Nearest racing point. Searched locally when a previous index is known."""
    n = len(line)
    idxs = range(n) if near is None else ((near + k) % n for k in range(-5, window))
    best, bd = 0, float("inf")
    for i in idxs:
        d = (line[i][0] - x) ** 2 + (line[i][1] - y) ** 2
        if d < bd:
            bd, best = d, i
    return best


def _cross_track_error(line, i, x, y) -> float:
    n = len(line)
    ax, ay = line[i][0], line[i][1]
    bx, by = line[(i + 1) % n][0], line[(i + 1) % n][1]
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy) or 1e-9
    return abs((x - ax) * dy - (y - ay) * dx) / L


# ---------------------------------------------------------------------------
# Policy family
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Policy:
    """Pure pursuit with three knobs. Each setting drives measurably differently."""
    lookahead: int          # racing points to look ahead. Low = twitchy, high = smooth
    speed_scale: float      # fraction of the racing line's prescribed speed
    lateral_bias: float     # metres offset from the line. Positive = outside

    @property
    def name(self) -> str:
        return f"la{self.lookahead}_sp{self.speed_scale:.2f}_off{self.lateral_bias:+.2f}"


def policy_grid() -> list[Policy]:
    """A spread of behaviours from cautious to reckless."""
    out = []
    for la in (3, 6, 10, 16):
        for sp in (0.45, 0.60, 0.75, 0.90, 1.00):
            for off in (-0.25, 0.0, 0.25):
                out.append(Policy(la, sp, off))
    return out


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class Lap:
    policy: Policy
    finished: bool
    steps: int
    lap_time: float                  # seconds; inf when the car did not finish
    params_trace: list[dict] = field(default_factory=list)


def _snap(actions, steer, speed) -> tuple[float, float]:
    """Snap a continuous command to the nearest discrete action."""
    return min(actions, key=lambda a: ((a[0] - steer) / 60.0) ** 2 + ((a[1] - speed) / 4.0) ** 2)


def simulate(line, actions, pol: Policy, track_width=TRACK_HALF_WIDTH * 2) -> Lap:
    n = len(line)
    total_len = sum(math.hypot(line[(i + 1) % n][0] - line[i][0],
                               line[(i + 1) % n][1] - line[i][1]) for i in range(n))

    # Start on the line, pointing along it.
    x, y = line[0][0], line[0][1]
    heading = math.degrees(math.atan2(line[1][1] - y, line[1][0] - x))
    speed = line[0][2] * pol.speed_scale
    idx = 0
    visited = 0
    trace: list[dict] = []

    for step in range(1, MAX_STEPS + 1):
        prev_idx = idx
        idx = _closest_index(line, x, y, near=idx)
        visited += (idx - prev_idx) % n

        # target a point ahead, offset laterally
        t = line[(idx + pol.lookahead) % n]
        tp = line[(idx + pol.lookahead - 1) % n]
        tn = line[(idx + pol.lookahead + 1) % n]
        ndx, ndy = -(tn[1] - tp[1]), (tn[0] - tp[0])
        nl = math.hypot(ndx, ndy) or 1e-9
        tx = t[0] + (ndx / nl) * pol.lateral_bias
        ty = t[1] + (ndy / nl) * pol.lateral_bias

        # pure pursuit steering
        bearing = math.degrees(math.atan2(ty - y, tx - x))
        err = (bearing - heading + 180.0) % 360.0 - 180.0
        steer_cmd = max(-MAX_STEER_DEG, min(MAX_STEER_DEG, err))
        speed_cmd = t[2] * pol.speed_scale

        steer, target_speed = _snap(actions, steer_cmd, speed_cmd)

        cte = _cross_track_error(line, idx, x, y)
        offtrack = cte > track_width / 2.0

        trace.append({
            "all_wheels_on_track": not offtrack,
            "x": x, "y": y,
            "closest_waypoints": [idx, (idx + 1) % n],
            "distance_from_center": cte,
            "is_crashed": False,
            "is_left_of_center": True,
            "is_offtrack": offtrack,
            "is_reversed": False,
            "heading": heading,
            "progress": min(100.0, visited / n * 100.0),
            "speed": speed,
            "steering_angle": steer,
            "steps": step,
            "track_length": total_len,
            "track_width": track_width,
            "waypoints": [[p[0], p[1]] for p in line],
        })

        if offtrack:
            return Lap(pol, False, step, float("inf"), trace)
        if visited >= n:
            trace[-1]["progress"] = 100.0
            return Lap(pol, True, step, step * DT, trace)

        # kinematic bicycle update
        speed += (target_speed - speed) * (DT / SPEED_TAU)
        heading += _yaw_rate_deg(speed, steer) * DT
        heading = (heading + 180.0) % 360.0 - 180.0
        x += speed * math.cos(math.radians(heading)) * DT
        y += speed * math.sin(math.radians(heading)) * DT

    return Lap(pol, False, MAX_STEPS, float("inf"), trace)


# ---------------------------------------------------------------------------
# The test: does the reward agree with the clock?
# ---------------------------------------------------------------------------

def score_lap(reward_fn, lap: Lap, mode: str = "total") -> float:
    """Score a lap.

    mode="total" is the DEFAULT and the correct one. Reinforcement learning
    maximises the CUMULATIVE reward over an episode, so that is the quantity
    that decides what policy you get.

    mode="mean" divides by the step count. This was the original default and it
    was wrong. Dividing by steps normalises away the step count, which hides the
    most common failure in a hand-written reward function: a positive per-step
    reward means a SLOWER lap has more steps and therefore earns more. The mean
    looks healthy while the sum is telling the car to crawl. See the Round 2
    retraction in the README.

    Reward functions print on every step; that output is discarded.
    """
    total, n = 0.0, 0
    with contextlib.redirect_stdout(io.StringIO()):
        for p in lap.params_trace:
            try:
                total += float(reward_fn(dict(p)))
            except Exception:
                pass  # a raising step contributes nothing, which is itself a signal
            n += 1
    if mode == "mean":
        return total / n if n else 0.0
    return total


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation. No numpy."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((rb[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da and db else 0.0
