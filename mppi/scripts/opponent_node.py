#!/usr/bin/env python3
"""Opponent car for head-to-head testing in the simulator (and later a second real car).

v0: pure pursuit on a raceline csv with a speed scale and a constant sideways offset. No JAX,
so several of them are cheap to run next to the ego MPPI.

    speed_scale     x the csv speed profile (0.6 = a slow car to pass on the straight)
    max_speed       cap [m/s]
    lateral_offset  [m] drive this far left (+) / right (-) of the raceline

The defending behavior of the final project (cover the inside before a corner, adjustable
aggressiveness) goes on top of this node.
"""
import os

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile

WB = 0.33  # wheelbase [m]


def load_line(path, ds=0.1):
    """closed raceline csv (x, y[, yaw, v]) -> (N, 4) [x, y, yaw, v], one point every ds"""
    data = np.loadtxt(path, delimiter=',', comments='#', ndmin=2)
    xy = data[:, :2]
    v = data[:, 3] if data.shape[1] >= 4 else np.full(len(xy), 2.0)
    if np.hypot(*(xy[0] - xy[-1])) < 1e-3:
        xy, v = xy[:-1], v[:-1]
    xy_c = np.vstack([xy, xy[:1]])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy_c, axis=0).T))]
    n = int(s[-1] / ds)
    s_new = np.linspace(0.0, s[-1], n, endpoint=False)
    x = np.interp(s_new, s, xy_c[:, 0])
    y = np.interp(s_new, s, xy_c[:, 1])
    v = np.interp(s_new, s, np.r_[v, v[:1]])
    yaw = np.arctan2(np.roll(y, -1) - np.roll(y, 1), np.roll(x, -1) - np.roll(x, 1))
    return np.c_[x, y, yaw, v]


class Opponent(Node):
    def __init__(self):
        super().__init__('opponent_node')
        self.declare_parameter('odom_topic', '/opp_racecar/odom')
        self.declare_parameter('drive_topic', '/opp_drive')
        self.declare_parameter('waypoint_file', 'race2_raceline.csv')
        self.declare_parameter('speed_scale', 0.6)
        self.declare_parameter('max_speed', 6.0)
        self.declare_parameter('lateral_offset', 0.0)
        self.declare_parameter('lookahead_min', 0.6)    # [m]
        self.declare_parameter('lookahead_time', 0.35)  # [s] lookahead = min + v * time
        self.declare_parameter('max_steer', 0.4189)
        self.declare_parameter('rate', 40.0)            # [Hz] command rate

        path = self.get_parameter('waypoint_file').value
        if not os.path.isabs(path):
            path = os.path.join(get_package_share_directory('mppi'), 'waypoints', path)
        self.wps = load_line(path)
        self.ds = 0.1
        self.pose = None  # (x, y, yaw, v)
        qos = QoSProfile(depth=1)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.odom_cb, qos)
        self.pub = self.create_publisher(AckermannDriveStamped, self.get_parameter('drive_topic').value, 1)
        self.create_timer(1.0 / float(self.get_parameter('rate').value), self.step)
        self.get_logger().info(
            f'opponent: {len(self.wps)} waypoints from {path}, '
            f'speed x{self.get_parameter("speed_scale").value}, '
            f'offset {self.get_parameter("lateral_offset").value} m')

    def odom_cb(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
        self.pose = (p.x, p.y, yaw, msg.twist.twist.linear.x)

    def step(self):
        if self.pose is None:
            return
        x, y, yaw, v = self.pose
        wps, n = self.wps, len(self.wps)
        offset = float(self.get_parameter('lateral_offset').value)
        i0 = int(np.argmin(np.hypot(wps[:, 0] - x, wps[:, 1] - y)))
        look = float(self.get_parameter('lookahead_min').value) \
            + max(v, 0.0) * float(self.get_parameter('lookahead_time').value)
        it = (i0 + int(round(look / self.ds))) % n
        # target point, moved sideways along the left normal of the line
        tx = wps[it, 0] - offset * np.sin(wps[it, 2])
        ty = wps[it, 1] + offset * np.cos(wps[it, 2])
        dx, dy = tx - x, ty - y
        y_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
        dist2 = max(dx * dx + dy * dy, 1e-6)
        max_steer = float(self.get_parameter('max_steer').value)
        steer = float(np.clip(np.arctan(2.0 * WB * y_local / dist2), -max_steer, max_steer))
        # speed: the slowest profile value over the next ~0.3 s, so the car brakes before a corner
        ahead = (i0 + np.arange(1 + int(max(v, 1.0) * 0.3 / self.ds))) % n
        speed = float(np.min(wps[ahead, 3])) * float(self.get_parameter('speed_scale').value)
        speed = min(speed, float(self.get_parameter('max_speed').value))

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Opponent()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
