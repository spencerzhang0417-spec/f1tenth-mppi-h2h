"""Real car on Levine 2nd floor (NOT the autograder launch):

    ros2 launch mppi levine_car_launch.py

Needs the car stack running first: VESC / ackermann_mux (/drive),
particle filter (/pf/pose/odom) with the 2nd-floor map, and hold the
deadman button on the joystick for autonomous driving.
Put levine_2nd.csv (x, y, yaw, v) in mppi/waypoints/ and rebuild.
"""
from launch import LaunchDescription
from launch_ros.actions import Node

EXECUTABLE = 'mppi_node.py'

PARAMETERS = {
    'track': 'levine',
    # real car: pose from /pf/pose/odom, speed from VESC /odom
    'use_sim': False,
    'odom_topic': '/pf/pose/odom',
    'speed_topic': '/odom',
    'drive_topic': '/drive',

    # car csv with x, y, yaw, v (same file as the MPC lab)
    'waypoint_file': 'levine_2nd.csv',
    # 'keep' = drive in the direction the csv was recorded
    # ('auto' would force counter-clockwise like the graded sim run)
    'waypoint_direction': 'keep',

    # csv speed is 0.5-1.0 m/s; x 5.0 = 2.5-5 m/s, then capped by max_speed.
    # With max_speed 2.0 almost the whole lap is 2 m/s (corners not slower): fine for a first run.
    # Later: raise max_speed and lower speed_scale so the csv's corner slow-down survives.
    'speed_scale': 5.0,
    'max_speed': 2.0,
    'accel_lead_time': 0.05,            # = DT; the VESC tracks speed well (0.19-0.21 was for the sim)

    # compute on the car: check "control step" / "max gap" in the log;
    # lower num_samples (512 -> 256) if max gap shows "<- hiccup"
    'num_samples': 512,
    'latency_comp': -1.0,               # -1 = automatic (measured control-step time), 0 = off
    'visualize': False,                 # rollouts in RViz are expensive on the car: debugging only

    # MPPI weights (defaults = the skeleton's MPPIConfig); now declared in the node, so they apply
    'Q':  [5.0, 5.0, 1.0, 1.0],         # state cost        [x, y, yaw, v]
    'Qf': [5.0, 5.0, 1.0, 1.0],         # final state cost  [x, y, yaw, v]
    'R':  [0.01, 0.01],                 # input cost        [accel, steering]
    # bigger steering Rd = smoother steering but slower to turn in (like Rdk in the MPC lab)
    'Rd': [0.01, 1.0],                  # input-change cost [accel, steering]
    'lambda': 0.01,                     # temperature: bigger = smoother / less greedy
    'u_std': [1.0, 0.1],                # sampling noise    [accel, steering]

    # obstacle avoidance from /scan stays off on the car for now
    'avoid_obstacles': False,
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