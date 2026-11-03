"""Race 2 on the real car: MPPI + obstacle avoidance from /scan (NOT the autograder launch).

    ros2 launch mppi race2_car_launch.py

Needs the car stack running first: VESC / ackermann_mux (/drive), particle filter
(/pf/pose/odom) with the race map, hold the deadman button. Put race2_raceline.csv in
mppi/waypoints/ and rebuild.
"""
from launch import LaunchDescription
from launch_ros.actions import Node

EXECUTABLE = 'mppi_node.py'

PARAMETERS = {
    'track': 'levine',
    'use_sim': False,
    'odom_topic': '/pf/pose/odom',
    'speed_topic': '/odom',
    'drive_topic': '/drive',

    'waypoint_file': 'race2_raceline_remapped.csv',  # x, y only -> speed from curvature (1.5-4 m/s)
    'waypoint_direction': 'keep',           # drive the csv in its recorded direction
    # curvature speed 1.5-4 m/s x 0.75 = 1.1-3 m/s; corners stay slower than straights.
    # the profile is built for 3 m/s^2 lateral accel, x 0.75^2 -> ~1.7 m/s^2 in the corners
    # (was 0.5 -> 0.75-2 m/s, ~0.75 m/s^2)
    'speed_scale': 0.75,
    # bag 2: contacts at 1.7-1.8 m/s. wobble_1102_1337 ran clean at 1.5 (commands hit the cap,
    # p95 lateral accel 1.8 m/s^2). max_speed alone above 2.0 does nothing: speed_scale caps v_ref
    'max_speed': 4.0,
    # the VESC servo range 0.15..0.85 (vesc.yaml, gain -1.2135, offset 0.52) only allows -0.272 (right)
    # .. +0.305 (left) rad; MPPI planned with +-0.419 and ran wide into the outer wall (wobble_1102_1542:
    # commands -0.35..-0.42 gave -0.275 measured, 23% of steps asked for more than the servo has)
    # vesc offset 0.54: the servo range allows -0.255 (right) .. +0.321 (left) rad
    'max_steer': 0.25,
    # wobble_1103_1556: both hairpins are LEFT turns and the car ran ~0.5-0.7 m wide out of them
    # into the obstacles behind the exit (stops at 75 s, 103 s, 203 s). With +-0.25 the MPPI mean
    # (an average of clipped samples) only reached +0.21..0.23 -> R ~1.3-1.5 m. The left side has
    # servo room up to 0.321, so use it there; the right side stays at the servo limit
    'max_steer_left': 0.32,
    'max_steer_right': 0.25,
    # speed command = v + a * lead when MPPI accelerates (braking keeps DT = 0.05 s). 0.05 asked for
    # <= 0.09 m/s over the current speed (p95) -> only ~0.7 m/s^2 (p95) in wobble_1102_1350 / 1431
    'accel_lead_time': 0.2,

    # compute: the log line shows "control step", "grid" and "bend" times.
    # control step must stay well below 50 ms; if not, lower num_samples to 256
    'num_samples': 512,
    'cold_iters': 3,                    # 10 cold-start iterations froze the car for ~1 s
    # bag wobble_1102_1322: pose age ~55 ms + steering -> IMU response ~140 ms. -1 (auto) only
    # covered the ~50 ms compute time, and the car swung 40-70 cm left-right on straights.
    # 0.2 missed that the PF heading itself lags the IMU by ~120 ms (all bags 1322-1507):
    # total ~0.12 + 0.05 + 0.15 = 0.32 s (sim at 1.85 m/s: 30 cm swing with 0.2, 5 cm with 0.32)
    'latency_comp': 0.32,
    # predict over that delay with the commands actually sent in the last 0.32 s, not the last one
    # held (closed-loop sim, box obstacle, 0.315 s delay: 2-3 of 6 passes hit with hold-last, 0 of
    # 12 with the history; straight-line swing 4.8 -> 3.6 cm)
    'latency_history': True,
    'visualize': False,

    # heading 3 / Rd steer 1.5 was tried and reverted: the under-steer at s 10.5-13.5 m in wobble_1102_1620
    # came from the obstacle cost reacting to floor returns (lidar tilted ~3-5 deg down), not from these weights
    'Q':  [5.0, 5.0, 1.0, 1.0],
    'Qf': [5.0, 5.0, 1.0, 1.0],
    'R':  [0.01, 0.01],
    # Rd: 0.01, 2.0, min speed 1.25
    'Rd': [0.01, 2.0],                  # smoother steering (the bag showed +-0.4 rad flips)
    'lambda': 0.01,                     # only used when ess_target is 0
    'ess_target': 0.1,                  # adaptive lambda: ~10% of the samples count (ESS ~51/512)
    'u_std': [1.0, 0.1],

    # ---- obstacle avoidance from /scan ----
    'avoid_obstacles': True,
    'scan_topic': '/scan',
    'lidar_offset': 0.27,               # laser ahead of the pose point [m]
    'steer_std': 0.15,
    # real car: ~10 Hz control + steering lag + localization noise -> bigger margins than in sim
    # (0.17 left 1.5 cm; in bag 2 the driven path was only 0.10-0.25 m from the obstacles)
    'obs_radius': 0.24,
    'obs_margin': 0.40,
    'slow_near_obstacles': True,
    'slow_min_factor': 0.3,
    'slow_front_angle': 0.8,            # +-46 deg front cone (contacts came in at +40..+60 deg)
    # full speed once the nearest return in that cone is >= this [m] (default 3.0). The factor
    # scales the profile before the ref is built, so it really sets the speed; walls at ~1.15 m in
    # the cone give ~0.5x with 3.0 (vs ~0.8x with 1.5). 3.0 after the crash in wobble_1102_1507,
    # 1.5 again together with the earlier detours (bend_lead_*)
    'slow_front_free': 1.5,
    'shift_ref': True,
    'bend_lookahead': 4.0,
    'bend_extra': 0.10,                 # detour keeps >= obs_radius + 0.10 = 0.34 m clearance
    # wobble_1102_1519: obstacles seen 2.6-2.9 m ahead, but the detour only started ~0.6 m before
    # them, i.e. within the ~0.35 s pose + compute + steering delay. Finish it v * 0.35 s + 0.2 m
    # early (~0.9 m at 2 m/s), and keep it through the unseen space behind the obstacle
    'bend_lead_time': 0.35,
    'bend_lead_dist': 0.2,
    'bend_hold_blind': True,
}


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mppi',
            executable=EXECUTABLE,
            output='screen',
            parameters=[PARAMETERS],
        ),
    ])
