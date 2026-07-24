import math

def reward_function(params):
    if params.get('is_offtrack') or not params.get('all_wheels_on_track'):
        return 1e-3

    progress = params['progress']
    steps = params['steps']
    speed = params['speed']
    steering = params['steering_angle']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    heading = params['heading']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']

    # Base time penalty: every step costs (prevents step farming)
    reward = -1.0

    # Centerline keeping (zero-centered, small)
    norm_dist = distance_from_center / (track_width * 0.5)
    reward -= 0.5 * norm_dist * norm_dist

    # Steering smoothness (zero-centered, small)
    reward -= 0.1 * min(abs(steering) / 30.0, 1.0)

    # Speed incentive (capped positive, but net per-step stays negative)
    reward += 0.3 * (speed / 4.0)

    # Heading alignment with track direction
    prev_point = waypoints[closest_waypoints[0]]
    next_point = waypoints[closest_waypoints[1]]
    track_direction = math.degrees(
        math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    )
    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff
    reward -= 0.3 * direction_diff / 180.0

    # Finish bonus encodes time directly:
    #   flat completion bonus (exceeds worst realistic accumulated penalty)
    #   + linear reward scaling with steps saved under target
    if progress >= 99.5:
        target_steps = 400
        margin = target_steps - steps
        reward += 300.0 + max(0.0, margin) * 20.0

    return float(reward)