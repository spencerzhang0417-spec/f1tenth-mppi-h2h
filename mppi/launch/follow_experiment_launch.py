"""Hallway experiment: a human drives the opponent, the ego car estimates its speed and follows at
a fixed gap (default 0.5 m bumper to bumper). Straight line only.

Terminal 1 (SICK LiDAR + VESC):
~/setup_ros2.bash sick

Terminal 2 (perception + opponent EKF + follow), sourced as in vision_lab car_observations_launch.py:
cd ~/f1tenth_ws
source /opt/ros/jazzy/setup.bash
source ~/f1tenth_ws/install/setup.bash

env PATH="/usr/bin:$PATH" \
    PYTHONPATH="$HOME/numpy1_fix:${PYTHONPATH:-}" \
    ros2 launch mppi follow_experiment_launch.py

The env prefix is required while perception is started here (lab 10 node: system cv2, NumPy 1.x).

The car drives only while the deadman is held: joy button 8, the left trigger of the 8BitDo
Ultimate 2C (deadman_button:=<index>; -1 = no deadman). The left bumper (teleop) still overrides.

Arguments (same prefix):
    ros2 launch mppi follow_experiment_launch.py                          # perception started here
    ros2 launch mppi follow_experiment_launch.py perception:=false        # lab 10 node started elsewhere
    ros2 launch mppi follow_experiment_launch.py max_speed:=1.5 gap_ref:=0.6
    ros2 launch mppi follow_experiment_launch.py truth_topic:=/opp/odom   # log the opponent's own speed

Needs the car stack first (VESC + ackermann_mux on /drive, LiDAR on /scan, joy on /joy).
Starts: lab 10 car_observations (perception:=true), opponent_ekf_node, follow_node.
Log: ~/follow_logs/follow_<date>_<time>.csv -> python3 scripts/plot_follow.py <csv>
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

FOLLOW = {
    'drive_topic': '/drive',
    'ego_axle_to_front': 0.45,     # measure: ego rear axle -> front bumper
    'opp_rear_to_center': 0.28,    # measure: opponent rear bumper -> center (keep = ekf.surface_offset)
    'min_gap': 0.2,
    'k_gap': 1.0,
    'k_close': 1.5,
    'k_rel': 0.6,
    'max_accel': 2.0,
    'max_decel': 4.0,
    'track_timeout': 0.3,
    'steer_mode': 'pursuit',
    'max_steer': 0.25,             # servo limit (car_launch.py)
    'laser_x': 0.27,
    'corridor_half_width': 0.2,
    'estop_gap': 0.15,
}


def generate_launch_description():
    lc = LaunchConfiguration
    args = [
        DeclareLaunchArgument('perception', default_value='true'),
        DeclareLaunchArgument('imu', default_value='vesc'),
        DeclareLaunchArgument('gap_ref', default_value='0.5'),
        DeclareLaunchArgument('max_speed', default_value='1.5', description='start slow, raise per run'),
        DeclareLaunchArgument('truth_topic', default_value='none',
                              description="opponent's own odometry, logged only ('none' = off)"),
        DeclareLaunchArgument('deadman_button', default_value='8',
                              description='joy button to hold while driving (8 = left trigger); -1 = off'),
    ]
    ekf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('mppi'), 'launch',
                                                   'opponent_ekf_launch.py')),
        launch_arguments={'perception': lc('perception'), 'imu': lc('imu'), 'pose_topic': 'none',
                          'motion': 'straight'}.items(),
    )
    follow = Node(
        package='mppi',
        executable='follow_node.py',
        output='screen',
        parameters=[FOLLOW, {
            'gap_ref': ParameterValue(lc('gap_ref'), value_type=float),
            'max_speed': ParameterValue(lc('max_speed'), value_type=float),
            'truth_topic': ParameterValue(lc('truth_topic'), value_type=str),
            'deadman_button': ParameterValue(lc('deadman_button'), value_type=int),
        }],
    )
    return LaunchDescription(args + [ekf, follow])
