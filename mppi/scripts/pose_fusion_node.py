#!/usr/bin/env python3
"""High-rate pose for the controller: particle-filter pose + VESC speed + IMU gyro.

    /pf/pose/odom      (16 Hz, ~0.125 s old)  \\
    /odom              (50 Hz, speed)          >-->  /fused/odom  (50 Hz, now)
    /sensors/imu/raw   (50 Hz, yaw rate)      /

Point the controller's odom_topic at /fused/odom and set its latency_comp to what is left to
compensate: control step + steering response (~0.045 + 0.125 = 0.17 s) instead of 0.32 s.

Why (bag wobble_1102_1939, scripts/check_pose_fusion.py): 0.19 s after a particle-filter pose
the heading predicted from the steering COMMANDS is off by 6.7 deg rms, from the measured yaw
rate by 3.0 deg (the particle filter's own noise is ~2 deg).

  lag         [s] age of a particle-filter pose when it arrives (calibrate with the check script)
  lever       [m] the particle filter's pose point ahead of the rear axle (0.27 = the laser)
  gyro_scale  rad/s per IMU unit: the VESC IMU reports deg/s -> pi/180
  rate        [Hz] output rate; the controller runs once per pose
  The gyro bias is measured whenever the car stands still.
All times are this node's receive times: the particle filter's stamps are its publish time and
say nothing about the age of the pose.
"""
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Imu

from pose_fusion import PoseFuser


class PoseFusion(Node):
    def __init__(self):
        super().__init__('pose_fusion_node')
        self.declare_parameter('pf_topic', '/pf/pose/odom')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('imu_topic', '/sensors/imu/raw')
        self.declare_parameter('out_topic', '/fused/odom')
        self.declare_parameter('lag', 0.125)
        self.declare_parameter('lever', 0.27)
        self.declare_parameter('gyro_scale', math.pi / 180.0)
        self.declare_parameter('blend', 1.0)  # < 1: take only this fraction of each correction
        self.declare_parameter('rate', 50.0)  # [Hz] poses published per second at most
        gp = lambda k: self.get_parameter(k).value
        self.fuser = PoseFuser(lag=float(gp('lag')), lever=float(gp('lever')), blend=float(gp('blend')))
        self.scale = float(gp('gyro_scale'))
        self.v, self.bias, self.n_still = 0.0, 0.0, 0
        self.n_pose, self.corr2, self.last_log = 0, 0.0, 0.0
        self.period, self.last_pub = 1.0 / float(gp('rate')), -1e9
        qos = QoSProfile(depth=5)
        self.create_subscription(Odometry, gp('pf_topic'), self.pf_cb, qos)
        self.create_subscription(Odometry, gp('odom_topic'), self.odom_cb, qos)
        self.create_subscription(Imu, gp('imu_topic'), self.imu_cb, qos)
        self.pub = self.create_publisher(Odometry, gp('out_topic'), 5)
        self.get_logger().info(f'pose fusion: {gp("pf_topic")} (lag {gp("lag")} s, lever {gp("lever")} m) '
                               f'+ {gp("odom_topic")} + {gp("imu_topic")} -> {gp("out_topic")}')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, msg):
        self.v = msg.twist.twist.linear.x

    def imu_cb(self, msg):
        raw = msg.angular_velocity.z
        if abs(self.v) < 0.02:  # standing: whatever the gyro shows is its bias
            self.n_still += 1
            if self.n_still > 25:
                self.bias += 0.02 * (raw - self.bias)
        else:
            self.n_still = 0
        t = self.now()
        w = self.scale * (raw - self.bias)
        self.fuser.add_motion(t, self.v, w)
        if t - self.last_pub < self.period - 2e-3:
            return
        pose = self.fuser.pose(t)
        if pose is None:
            return
        self.last_pub = t
        out = Odometry()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'map'
        out.pose.pose.position.x, out.pose.pose.position.y = float(pose[0]), float(pose[1])
        out.pose.pose.orientation.z = math.sin(0.5 * pose[2])
        out.pose.pose.orientation.w = math.cos(0.5 * pose[2])
        out.twist.twist.linear.x = float(self.v)
        out.twist.twist.angular.z = float(w)
        self.pub.publish(out)

    def pf_cb(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
        t = self.now()
        self.fuser.add_pose(t, p.x, p.y, yaw)
        c = self.fuser.last_correction
        self.n_pose += 1
        self.corr2 += c[0] ** 2 + c[1] ** 2
        if t - self.last_log > 5.0 and self.n_pose > 1:
            self.get_logger().info(
                f'correction per particle-filter pose {100 * math.sqrt(self.corr2 / self.n_pose):.1f} cm rms '
                f'over {self.n_pose} poses, gyro bias {self.bias:+.3f}, v {self.v:.2f} m/s')
            self.last_log, self.n_pose, self.corr2 = t, 0, 0.0


def main(args=None):
    rclpy.init(args=args)
    node = PoseFusion()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
