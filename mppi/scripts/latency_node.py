#!/usr/bin/env python3
"""Puts the real car's delays between the simulator and the controller.

The simulator hands the controller a perfect pose with no delay and applies its commands at once.
On the car the pose is ~0.125 s old when it arrives, a control step takes ~0.045 s and the
steering needs ~0.125 s to act: ~0.3 s in total. A controller tuned without them is tuned for a
different car. Numbers from bag wobble_1102_1939 (race 2 track, 2 m/s average):
  laser 15.0 Hz, particle-filter pose 16.4 Hz, commands 16.0 Hz (494 of 1910 poses got none)
  steering command -> gyro yaw rate           125 ms (95-155)
  gyro yaw rate -> particle-filter yaw rate   125 ms (90-160)
  control step 45 ms: scan grid + detour 17, MPPI 15, warm-start shift 6, publish 4.5

    simulator /ego_racecar/odom --(pose_delay, pose_rate)--> /sim_delayed/odom  --> controller
    simulator /scan             --(scan_delay)------------> /sim_delayed/scan  --> controller
    controller /mppi/drive      --(actuation_delay)-------> /drive             --> simulator
    simulator /ego_racecar/odom --(50 Hz, no delay)-------> /sim/vesc_odom (speed only)
                                                            /sim/imu (yaw rate in deg/s, as the VESC IMU)
The last two are the car's fast sensors, for pose_fusion_node.py.

The delayed pose is stamped with the time it is handed over, as on the car: the particle filter
there stamps its pose when it publishes (stamp age 1.5 ms in the bag), not with the scan it was
computed from, so the stamp does not tell how old the pose really is. Scans keep their stamps.
  pose_delay       [s] particle filter + transport
  pose_rate        [Hz] poses per second handed to the controller (it runs once per pose)
  scan_delay       [s] the laser itself is nearly fresh: the pose is OLDER than the newest scan
  scan_rate        [Hz] scans per second (the car's laser: 15)
  actuation_delay  [s] compute time of the car's computer that this laptop does not need, plus
                   the steering / motor response
"""
from collections import deque

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan


class Latency(Node):
    def __init__(self):
        super().__init__('latency_node')
        self.declare_parameter('pose_delay', 0.12)
        self.declare_parameter('pose_rate', 16.0)
        self.declare_parameter('scan_delay', 0.02)
        self.declare_parameter('scan_rate', 15.0)
        self.declare_parameter('actuation_delay', 0.17)
        self.declare_parameter('odom_in', '/ego_racecar/odom')
        self.declare_parameter('odom_out', '/sim_delayed/odom')
        self.declare_parameter('scan_in', '/scan')
        self.declare_parameter('scan_out', '/sim_delayed/scan')
        self.declare_parameter('drive_in', '/mppi/drive')
        self.declare_parameter('drive_out', '/drive')
        gp = lambda k: self.get_parameter(k).value
        self.queues = []
        self._relay(Odometry, gp('odom_in'), gp('odom_out'), float(gp('pose_delay')), float(gp('pose_rate')),
                    restamp=True)
        self._relay(LaserScan, gp('scan_in'), gp('scan_out'), float(gp('scan_delay')), float(gp('scan_rate')))
        self._relay(AckermannDriveStamped, gp('drive_in'), gp('drive_out'), float(gp('actuation_delay')), 0.0)
        # the car's fast sensors, from the simulator's true motion
        self.vesc_pub = self.create_publisher(Odometry, '/sim/vesc_odom', 10)
        self.imu_pub = self.create_publisher(Imu, '/sim/imu', 10)
        self.fast_last = -1e9
        self.create_subscription(Odometry, gp('odom_in'), self.fast_cb, 10)
        self.create_timer(0.002, self.release)
        self.get_logger().info(
            f'delays: pose {gp("pose_delay")} s at {gp("pose_rate")} Hz, scan {gp("scan_delay")} s at {gp("scan_rate")} Hz, '
            f'actuation {gp("actuation_delay")} s')

    def _relay(self, msg_type, topic_in, topic_out, delay, rate, restamp=False):
        """forward topic_in to topic_out `delay` s later; rate > 0 keeps at most that many per second"""
        q = {'buf': deque(), 'pub': self.create_publisher(msg_type, topic_out, 10),
             'delay': delay, 'period': 1.0 / rate if rate > 0.0 else 0.0, 'last': -1e9,
             'restamp': restamp}

        def cb(msg, q=q):
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - q['last'] < q['period'] - 1e-4:
                return
            q['last'] = now
            q['buf'].append((now + q['delay'], msg))

        self.create_subscription(msg_type, topic_in, cb, 10)
        self.queues.append(q)

    def fast_cb(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.fast_last < 0.02 - 1e-4:
            return
        self.fast_last = now
        od = Odometry()
        od.header.stamp = self.get_clock().now().to_msg()
        od.twist.twist.linear.x = msg.twist.twist.linear.x
        self.vesc_pub.publish(od)
        im = Imu()
        im.header.stamp = od.header.stamp
        im.angular_velocity.z = msg.twist.twist.angular.z * 57.29578
        self.imu_pub.publish(im)

    def release(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        for q in self.queues:
            while q['buf'] and q['buf'][0][0] <= now:
                msg = q['buf'].popleft()[1]
                if q['restamp']:
                    msg.header.stamp = self.get_clock().now().to_msg()
                q['pub'].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Latency()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
