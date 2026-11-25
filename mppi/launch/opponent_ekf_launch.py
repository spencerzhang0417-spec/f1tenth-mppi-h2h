"""Opponent pose and speed on the real car: lab 10 car observations + VESC speed + gyro -> EKF.

    ros2 launch mppi opponent_ekf_launch.py                     # gyro: VESC IMU (/sensors/imu/raw)
    ros2 launch mppi opponent_ekf_launch.py imu:=camera         # gyro: RealSense D435i
    ros2 launch mppi opponent_ekf_launch.py perception:=true    # also start lab 10 car_observations
    ros2 launch mppi opponent_ekf_launch.py pose_topic:=/fused/odom pose_lag:=0.0   # map output from pose_fusion
    ros2 launch mppi opponent_ekf_launch.py pose_topic:=none     # no localization (hallway)

Needs the car stack (LiDAR /scan, VESC /odom and /sensors/imu/raw) and the lab 10 perception node
publishing /perception/car_observations (package vision_lab, branch car/jetson):
    ros2 launch vision_lab car_observations_launch.py           # or perception:=true here
The detector runs there, not here; this package only needs car_observation_msgs.

Outputs: /opponent/odom (ego frame), /opponent/odom_world (map, from pose_topic),
/opponent/observations_ego (the cleaned raw points, PoseArray, for Foxglove / RViz).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

EXECUTABLE = 'opponent_ekf_node.py'

PARAMETERS = {
    'observations_topic': '/perception/car_observations',
    'odom_topic': '/odom',

    # ---- observation cleaning (before the filter)
    'min_confidence': 0.5,           # lab 10 publishes >= its own threshold (0.5); raise to be stricter
    'min_range': 0.2,
    'max_range': 8.0,
    'max_bearing': 0.7,              # camera view +-35 deg
    'max_image_scan_skew': 0.1,      # [s] the camera picked the beam for an older / newer car position

    # ---- geometry
    'laser_x': 0.27,                 # LiDAR ahead of the rear axle (car_launch lidar_offset)

    # ---- EKF
    'ekf.surface_offset': 0.28,      # rear box -> car center (mppi_node h2h: depth 0.05 + center_ahead 0.23)
    'ekf.opp_accel_std': 2.0,        # larger: follows braking faster; smaller: smoother speed
    'ekf.opp_yaw_accel_std': 1.5,
    'ekf.speed_std': 0.15,
    'ekf.yaw_rate_std': 0.03,
    'ekf.range_std': 0.05,           # one LiDAR beam on the car surface
    'ekf.bearing_std': 0.03,         # the beam nearest the box center column: +-6 px ~ 0.01 rad, + car width
    'ekf.gate': 11.8,                # chi^2 2 dof, 99.7 %
    'ekf.confirm_hits': 3,           # gated updates within confirm_window before /opponent/odom is published
    'ekf.confirm_window': 0.5,
    'ekf.reinit_misses': 5,
    'ekf.lost_timeout': 1.0,
    'ekf.history': 1.0,              # [s] how late an observation may arrive
    'publish_rate': 50.0,
}

# motion:=straight (hallway experiments): the opponent drives straight along the ego's axis. Its heading
# cannot be observed while it stands still and otherwise wanders (seen: 120 deg after 3 s stopped), so
# the speed it then gains is put along the wrong direction. Pinning the heading and a smaller
# acceleration noise halve the speed error in the straight-line simulation (0.27 -> 0.14-0.17 m/s rms).
MOTION = {
    'race': {},
    'straight': {
        'ekf.opp_heading_std': 0.01,
        'ekf.opp_yaw_accel_std': 0.05,
        'ekf.init_yaw_rate_std': 0.1,
        'ekf.init_heading_std': 0.1,
        'ekf.opp_accel_std': 1.5,    # 1.0: smoother still, but slower to see the human brake
    },
}

# gyro source: (topic, world up in the IMU frame, rad/s per unit)
IMU = {
    'vesc': ('/sensors/imu/raw', [0.0, 0.0, 1.0], 0.017453292519943295),   # deg/s (pose_fusion_node)
    'camera': ('/camera/imu', [0.0, -1.0, 0.0], 1.0),                     # realsense_imu_node.py
}


def perception(context):
    if LaunchConfiguration('perception').perform(context).lower() not in ('true', '1'):
        return []
    path = os.path.join(get_package_share_directory('vision_lab'), 'launch', 'car_observations_launch.py')
    return [IncludeLaunchDescription(PythonLaunchDescriptionSource(path))]


def generate_launch_description():
    lc = LaunchConfiguration
    args = [
        DeclareLaunchArgument('imu', default_value='vesc', choices=list(IMU)),
        DeclareLaunchArgument('motion', default_value='race', choices=list(MOTION),
                              description='straight: the opponent drives straight (hallway)'),
        DeclareLaunchArgument('perception', default_value='false',
                              description='start lab 10 car_observations_launch.py (package vision_lab)'),
        DeclareLaunchArgument('pose_topic', default_value='/pf/pose/odom',
                              description="ego pose for /opponent/odom_world; 'none' = off"),
        DeclareLaunchArgument('pose_lag', default_value='0.125'),
    ]
    nodes = []
    for name, (topic, up, scale) in IMU.items():
        for motion, overrides in MOTION.items():
            nodes.append(Node(
                package='mppi',
                executable=EXECUTABLE,
                output='screen',
                parameters=[PARAMETERS, overrides, {
                    'imu_topic': topic,
                    'imu_up_axis': up,
                    'gyro_scale': scale,
                    'pose_topic': ParameterValue(lc('pose_topic'), value_type=str),
                    'pose_lag': ParameterValue(lc('pose_lag'), value_type=float),
                }],
                condition=IfCondition(PythonExpression(
                    ["'", lc('imu'), f"' == '{name}' and '", lc('motion'), f"' == '{motion}'"])),
            ))
    nodes.append(Node(
        package='mppi',
        executable='realsense_imu_node.py',
        output='screen',
        condition=LaunchConfigurationEquals('imu', 'camera'),
    ))
    return LaunchDescription(args + nodes + [OpaqueFunction(function=perception)])
