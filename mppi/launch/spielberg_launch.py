"""What the autograder runs on Spielberg (one lap):

    ros2 launch mppi spielberg_launch.py

Everything the graded run needs goes here: the nodes to start, the waypoints
to follow and the parameter values that suit Spielberg (levine_launch.py is
Levine's). Start your own nodes only: the simulator is already running.
"""
from launch import LaunchDescription
from launch_ros.actions import Node

# scripts/mppi_node.py
EXECUTABLE = 'mppi_node.py'

# `track` tells the skeleton which waypoints to load; add this track's values
# for the other parameters your node declares, e.g. 'max_speed': 6.0, or give
# it a full .yaml config file
PARAMETERS = {
    'track': 'spielberg',        # loads waypoints/spielberg.csv
    'use_sim': True,             # simulator
    'odom_topic': '/ego_racecar/odom',
    'drive_topic': '/drive',
    'waypoint_file': '',         # '' -> waypoints/spielberg.csv
    'speed_scale': 1.0,          # spielberg.csv already goes up to 8 m/s; 1.2 hit walls in testing
    'max_speed': 9.0,           # above the csv's 8 m/s so the speed profile is not cut
    'num_samples': 988,         # T
    'accel_lead_time': 0.18,     # sim speed controller lag (Lab 7 value); README formula = 0.05
    'visualize': True,          
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