"""Levine with obstacles (levine_obs): MPPI + obstacle avoidance from /scan.
Not graded. Switch the simulator map to levine_obs first, then:

    ros2 launch mppi levine_obs_launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node

EXECUTABLE = 'mppi_node.py'

PARAMETERS = {
    'track': 'levine',          # same centerline frame for every levine map
    'use_sim': True,
    'odom_topic': '/ego_racecar/odom',
    'drive_topic': '/drive',
    'waypoint_file': 'levine_outer.csv',  # 4 columns: x, y, yaw, v
    'speed_scale': 0.5,         # slow first; raise once the gaps are passed reliably
    'max_speed': 9.0,
    'num_samples': 512,
    'accel_lead_time': 0.19,
    'visualize': True,          # watch the rollouts in RViz

    # ---- obstacle avoidance from /scan ----
    'avoid_obstacles': True,    # subscribe /scan, build the local grid every step
    'scan_topic': '/scan',
    'lidar_offset': 0.0,        # = scan_distance_to_base_link in sim.yaml
    'steer_std': 0.2,           # wider steering samples so detours get sampled
    'obs_radius': 0.17,         # 5 footprint circles; 0.17 = car half width + 1.5 cm
    'obs_margin': 0.30,         # soft penalty starts at this clearance

    # ---- slow down near obstacles ----
    'slow_near_obstacles': True,
    'slow_min_factor': 0.12,    # reference speed x 0.12 when squeezing through a gap

    # ---- move the reference out of obstacles (otherwise MPPI stops in front of them) ----
    'shift_ref': True,
    'bend_lookahead': 4.0,      # [m] of path the detour is planned over
    'bend_extra': 0.03,         # [m] detour keeps obs_radius + this clearance
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