#!/usr/bin/env python3
"""Puts the simulated car against a wall and measures whether it gets going again.

    ros2 launch mppi h2h_sim_launch.py num_agents:=1 latency:=false pace:=race raceline:=race2_raceline_opt.csv
    python3 scripts/stuck_test.py --raceline mppi/waypoints/race2_raceline_opt.csv

Each case moves the car (through /initialpose, the simulator's reset) to station s of the
raceline, d m to the left of it, with the nose turned by dyaw rad: stopped, pointing at a wall, the
state the car was left in after a pass that went wrong. Reported per case: the time until it
drives faster than 1.5 m/s again, and wall contacts on the way.
"""
import argparse
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool

CASES = [(2.1, 0.45, 0.7), (13.1, -0.56, -0.6), (19.0, -0.25, -0.9), (10.0, 0.8, 1.2),
         (5.6, -0.6, -1.0), (15.0, 0.4, 0.9), (22.0, 0.6, 1.57), (8.5, -0.25, -1.0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--raceline', required=True)
    ap.add_argument('--timeout', type=float, default=12.0, help='[s] per case')
    args = ap.parse_args()
    xy = np.loadtxt(args.raceline, delimiter=',', comments='#', ndmin=2)[:, :3]
    seg = np.hypot(*np.diff(np.vstack([xy[:, :2], xy[:1, :2]]), axis=0).T)
    s_wp = np.r_[0.0, np.cumsum(seg)[:-1]]

    rclpy.init()
    node = Node('stuck_test')
    state = {'v': 0.0, 'hits': 0, 'in': False, 'n': 0}

    def odom(msg):
        state['v'], state['n'] = msg.twist.twist.linear.x, state['n'] + 1

    def coll(msg):
        if msg.data and not state['in']:
            state['hits'] += 1
        state['in'] = bool(msg.data)

    node.create_subscription(Odometry, '/ego_racecar/odom', odom, 10)
    node.create_subscription(Bool, '/ego_racecar/collision', coll, 10)
    pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 1)

    def spin(seconds, until=lambda: False):
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end and not until():
            rclpy.spin_once(node, timeout_sec=0.05)
        return until()

    spin(60.0, lambda: state['n'] > 0)
    spin(8.0)   # let it drive first
    ok = 0
    for s, d, dyaw in CASES:
        i = int(np.argmin(np.abs(s_wp - s)))
        x, y, yaw = xy[i]
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(x - d * np.sin(yaw))
        msg.pose.pose.position.y = float(y + d * np.cos(yaw))
        msg.pose.pose.orientation.z = float(np.sin(0.5 * (yaw + dyaw)))
        msg.pose.pose.orientation.w = float(np.cos(0.5 * (yaw + dyaw)))
        pub.publish(msg)
        spin(0.5)
        state['hits'], t0 = 0, time.monotonic()
        free = spin(args.timeout, lambda: state['v'] > 1.5)
        took = time.monotonic() - t0
        ok += bool(free)
        print(f's {s:5.1f} m, d {d:+.2f} m, nose {np.degrees(dyaw):+4.0f} deg: '
              + (f'driving again after {took:4.1f} s' if free else f'STILL STUCK after {took:.0f} s')
              + f', wall contacts {state["hits"]}', flush=True)
        spin(3.0)
    print(f'{ok} of {len(CASES)} cases recovered')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
