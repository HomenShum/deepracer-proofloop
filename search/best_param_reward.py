"""Generated reward function. Genome: {'w_const': -1.95643, 'w_dist': 0.54787, 'w_speed': 2.42792, 'w_align': 0.0, 'w_finish': 2986.09438, 'd0': 0.37919}"""
import math


def reward_function(params):
    W_CONST  = -1.95643
    W_DIST   = 0.54787
    W_SPEED  = 2.42792
    W_ALIGN  = 0.0
    W_FINISH = 2986.09438
    D0       = 0.37919

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
