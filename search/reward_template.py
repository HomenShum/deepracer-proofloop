"""
Turn a reward-design genome into real reward function source.

The output is a normal DeepRacer reward_function(params) file, directly
comparable to the twelve human-written ones. That matters: the search should
produce the same kind of artefact a person would, not a private closure that
cannot be inspected or shipped.

The design space is chosen from what the twelve human functions got wrong:

  w_const     constant per step. NEGATIVE is a time penalty, which is the term
              every human function was missing. Seven of twelve were
              anti-correlated with speed because their per-step reward was
              positive, so a slower lap earned more.
  w_dist      reward for holding the racing line
  w_speed     reward for raw speed
  w_align     reward for pointing the right way
  w_finish    one-off bonus at progress == 100
  d0          distance scale at which the line reward decays to zero

Off-track is always fail-closed at 1e-3. That is not searchable. It is the one
gate that is not up for negotiation.
"""

from __future__ import annotations

GENOME = ["w_const", "w_dist", "w_speed", "w_align", "w_finish", "d0"]

# w_const must be able to DOMINATE the shaping terms, otherwise the design
# space cannot express a net-negative per-step reward and every candidate is a
# step farmer by construction. The first version capped it at -1.0 against
# shaping terms summing to +8.0, so a deliberate "time penalty" design still
# scored rho(steps) = +1.000. The range below lets the search find the
# standard time-minimisation formulation: negative per step, large bonus at
# the finish.
# w_finish must be able to exceed the ACCUMULATED time penalty, or the optimal
# policy is to crash on step one and stop paying it. A 275-step lap at a net
# -1.65 per step costs about -454, so a bonus capped at 800 was already close
# to the edge and a stronger penalty went past it. The observed failure:
# w_const=-4.0 scored rho(speed)=+1.000, exactly right, and still trained to a
# DNF because ending the episode early beat finishing.
#
# The valid region is a narrow band. Too positive per step and the car farms
# steps by driving slowly. Too negative and it crashes to stop the bleeding.
# All twelve human functions sat on the step-farming side of it.
LO = [-6.0, 0.0, 0.0, 0.0, 0.0, 0.05]
HI = [0.5, 3.0, 3.0, 2.0, 4000.0, 0.60]
INIT_MU = [-1.5, 1.0, 1.0, 0.5, 1200.0, 0.25]
INIT_SIGMA = [2.0, 0.9, 0.9, 0.6, 1200.0, 0.15]

TEMPLATE = '''"""Generated reward function. Genome: {genome!r}"""
import math


def reward_function(params):
    W_CONST  = {w_const!r}
    W_DIST   = {w_dist!r}
    W_SPEED  = {w_speed!r}
    W_ALIGN  = {w_align!r}
    W_FINISH = {w_finish!r}
    D0       = {d0!r}

    if params.get("is_offtrack") or not params.get("all_wheels_on_track", True):
        return 1e-3
    if params.get("is_reversed"):
        return 1e-3

    dist = float(params.get("distance_from_center", 0.0))
    speed = float(params.get("speed", 0.0))
    progress = float(params.get("progress", 0.0))
    heading = float(params.get("heading", 0.0))
    waypoints = params.get("waypoints") or []
    closest = params.get("closest_waypoints") or [0, 0]

    reward = W_CONST

    if D0 > 0:
        reward += W_DIST * max(0.0, 1.0 - dist / D0)

    reward += W_SPEED * (speed / 4.0)

    if len(waypoints) > 1:
        a = waypoints[closest[0] % len(waypoints)]
        b = waypoints[closest[1] % len(waypoints)]
        track_dir = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        diff = abs(track_dir - heading)
        if diff > 180:
            diff = 360 - diff
        reward += W_ALIGN * max(0.0, 1.0 - diff / 30.0)

    if progress >= 100.0:
        reward += W_FINISH

    return float(max(reward, -10.0))
'''


def render(genome) -> str:
    g = {k: round(float(v), 5) for k, v in zip(GENOME, genome)}
    return TEMPLATE.format(genome=g, **g)


def clip(genome):
    return [max(LO[i], min(HI[i], float(v))) for i, v in enumerate(genome)]
