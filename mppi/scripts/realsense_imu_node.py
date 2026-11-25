#!/usr/bin/env python3
"""Publish the RealSense D435i gyro as sensor_msgs/Imu on /camera/imu (pyrealsense2).

Lab 10 reads the color stream through V4L2 (camera_node.py), so realsense2_camera is not
running; this node opens only the motion module. Alternative: realsense2_camera with
enable_color:=false enable_depth:=false enable_gyro:=true, and point opponent_ekf_node imu_topic at
its gyro topic.

Axes are the RealSense IMU axes (x right, y down, z forward, like the optical frame); the EKF
node maps them to yaw rate with imu_up_axis. Only angular_velocity is filled (orientation and
linear acceleration covariances are -1 = not provided). Stamped with this node's clock.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu


class RealSenseImuNode(Node):
    def __init__(self):
        super().__init__('realsense_imu_node')
        self.declare_parameter('topic', '/camera/imu')
        self.declare_parameter('frame_id', 'camera_imu_optical_frame')
        self.declare_parameter('gyro_fps', 200)  # D435i: 200 or 400
        import pyrealsense2 as rs

        self.frame_id = self.get_parameter('frame_id').value
        self.pub = self.create_publisher(Imu, self.get_parameter('topic').value, 50)
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f,
                          int(self.get_parameter('gyro_fps').value))
        self.pipe.start(cfg, self.on_frame)  # callback runs on a librealsense thread
        self.get_logger().info(f'RealSense gyro -> {self.get_parameter('topic').value}')

    def on_frame(self, frame):
        if not frame.is_motion_frame():
            return
        d = frame.as_motion_frame().get_motion_data()
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = d.x, d.y, d.z
        msg.orientation_covariance[0] = -1.0
        msg.linear_acceleration_covariance[0] = -1.0
        self.pub.publish(msg)

    def destroy_node(self):
        try:
            self.pipe.stop()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RealSenseImuNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
