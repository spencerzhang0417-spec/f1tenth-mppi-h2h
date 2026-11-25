#!/usr/bin/env python3
"""Opponent pose and speed from the lab 10 car observations, the VESC and a gyro.

    /perception/car_observations  (lab 10 vision_lab car_observations.py) --\\
    /odom              (VESC speed)                                          >-- EKF --> /opponent/odom
    /sensors/imu/raw   (gyro)                                               /            /opponent/odom_world
    /pf/pose/odom      (optional: ego pose, for the map-frame output)    --'

The detector is NOT in this node: lab 10 runs it and publishes, per LiDAR scan, the LiDAR surface
point of every detected car (x forward, y left, in the LiDAR frame) with its confidence and its
beam time (header.stamp + time_offset). This node only cleans and filters those points.

Cleaning, before the filter (per observation):
    - not finite, closer than min_range or further than max_range   -> dropped
    - confidence < min_confidence                                    -> dropped
    - |bearing| > max_bearing (outside the camera view: not possible) -> dropped
    - |image_stamp - beam time| > max_image_scan_skew                  -> dropped (the camera
      picked the beam for a car that has since moved: the beam likely hit the background)
In the filter (opponent_ekf.py): chi^2 gating against the prediction, track confirmation
(confirm_hits within confirm_window) and re-initialization. An empty array is a valid frame
with no car; the track times out after ekf.lost_timeout without an accepted point.

/opponent/odom (nav_msgs/Odometry), published at publish_rate while a CONFIRMED car is tracked:
    header.frame_id   base_frame: the ego rear axle, x forward, y left
    pose              the opponent's CENTER and heading relative to the ego, with covariance
    twist             in the opponent's own frame (child_frame_id): linear.x = its speed over
                      ground [m/s], angular.z = its yaw rate [rad/s], with covariance
/opponent/odom_world: the same in world_frame, from the ego pose on pose_topic (off when '').
/opponent/observations_ego (geometry_msgs/PoseArray): the cleaned raw points in base_frame, for
debugging the filter against its input in Foxglove / RViz.

Times: the LiDAR beam time for observations (the LiDAR driver's stamp, same clock as this node);
receive time for /odom and the gyro. The ego pose on pose_topic is taken as pose_lag old when it
arrives (like pose_fusion_node: 0.125 for /pf/pose/odom, 0 for /fused/odom), and its point is
pose_lever ahead of the rear axle.

Gyro: yaw rate = gyro_scale * (angular_velocity . imu_up_axis) - bias, imu_up_axis = world up
in the IMU frame. VESC IMU (deg/s): [0, 0, 1] and pi/180. RealSense D435i (x right, y down,
z forward), camera level: [0, -1, 0] and 1. The bias is learned while the car stands still.
"""
import dataclasses
import math
import time

import numpy as np
import rclpy
from car_observation_msgs.msg import CarObservationArray
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu

from opponent_ekf import EKFParams, OpponentEKF, PHI, PX, PY, VO, WO

RESULTS = ('init', 'update', 'reinit', 'reject', 'none', 'stale')
DROPS = ('range', 'confidence', 'bearing', 'skew')


def stamp_sec(stamp):
    return stamp.sec + 1e-9 * stamp.nanosec


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2))


class OpponentEKFNode(Node):
    def __init__(self):
        super().__init__('opponent_ekf_node')
        defaults = {
            # topics
            'observations_topic': '/perception/car_observations',
            'odom_topic': '/odom',
            'imu_topic': '/sensors/imu/raw',
            'pose_topic': '',             # '' or 'none' = no map-frame output
            'output_topic': '/opponent/odom',
            'world_output_topic': '/opponent/odom_world',
            'debug_points_topic': '/opponent/observations_ego',   # '' = off
            'base_frame': 'base_link',
            'world_frame': 'map',
            'child_frame': 'opponent',
            'publish_rate': 50.0,
            # observation cleaning
            'min_confidence': 0.5,
            'min_range': 0.2,             # [m] LiDAR range of the surface point
            'max_range': 8.0,
            'max_bearing': 0.7,           # [rad] D435i color: +-35 deg horizontal
            'max_image_scan_skew': 0.1,   # [s] |image time - beam time|; <= 0 = off
            # LiDAR pose on the ego (ahead of the rear axle)
            'laser_x': 0.27,
            'laser_y': 0.0,
            'laser_yaw': 0.0,
            # gyro
            'imu_up_axis': [0.0, 0.0, 1.0],
            'gyro_scale': math.pi / 180.0,
            'estimate_gyro_bias': True,
            # ego pose for the world output
            'pose_lag': 0.125,
            'pose_lever': 0.27,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        ekf_defaults = EKFParams()
        for f in dataclasses.fields(EKFParams):
            self.declare_parameter(f'ekf.{f.name}', getattr(ekf_defaults, f.name))
        gp = lambda k: self.get_parameter(k).value
        pose_topic = '' if gp('pose_topic') in ('', 'none') else gp('pose_topic')

        # ---- filter
        self.ekf = OpponentEKF(EKFParams(**{f.name: type(getattr(ekf_defaults, f.name))(gp(f'ekf.{f.name}'))
                                            for f in dataclasses.fields(EKFParams)}))
        self.laser_xy = np.array([float(gp('laser_x')), float(gp('laser_y'))])
        c, s = math.cos(float(gp('laser_yaw'))), math.sin(float(gp('laser_yaw')))
        self.laser_rot = np.array([[c, -s], [s, c]])
        up = np.asarray(gp('imu_up_axis'), dtype=np.float64)
        self.up = up / np.linalg.norm(up)
        self.gyro_scale = float(gp('gyro_scale'))
        self.estimate_bias = bool(gp('estimate_gyro_bias'))
        self.v, self.t_still, self.bias = 0.0, None, 0.0
        self.pose = None  # (t, x, y, yaw) of the ego rear axle
        self.pose_lag, self.pose_lever = float(gp('pose_lag')), float(gp('pose_lever'))
        self.base_frame, self.world_frame, self.child_frame = gp('base_frame'), gp('world_frame'), gp('child_frame')

        # ---- cleaning
        self.min_conf = float(gp('min_confidence'))
        self.min_range, self.max_range = float(gp('min_range')), float(gp('max_range'))
        self.max_bearing = float(gp('max_bearing'))
        self.max_skew = float(gp('max_image_scan_skew'))

        # ---- stats for the log line
        self.counts = dict.fromkeys(RESULTS, 0)
        self.drops = dict.fromkeys(DROPS, 0)
        self.n_msgs, self.n_obs, self.latency, self.t_log = 0, 0, 0.0, time.monotonic()

        # ---- ROS
        self.create_subscription(CarObservationArray, gp('observations_topic'), self.observations_cb, 10)
        self.create_subscription(Odometry, gp('odom_topic'), self.odom_cb, 20)
        self.create_subscription(Imu, gp('imu_topic'), self.imu_cb, qos_profile_sensor_data)
        if pose_topic:
            self.create_subscription(Odometry, pose_topic, self.pose_cb, 5)
        self.pub = self.create_publisher(Odometry, gp('output_topic'), 10)
        self.world_pub = self.create_publisher(Odometry, gp('world_output_topic'), 10) if pose_topic else None
        self.points_pub = self.create_publisher(PoseArray, gp('debug_points_topic'), 10) \
            if gp('debug_points_topic') else None
        self.create_timer(1.0 / float(gp('publish_rate')), self.publish)
        self.get_logger().info(
            f'opponent EKF: {gp("observations_topic")} + {gp("odom_topic")} + {gp("imu_topic")} '
            f'(up {self.up.round(3).tolist()}, scale {self.gyro_scale:.5g}) -> {gp("output_topic")}'
            + (f' and {gp("world_output_topic")} via {pose_topic}' if pose_topic else ''))

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ ego motion
    def odom_cb(self, msg):
        t = self.now()
        self.v = msg.twist.twist.linear.x
        if abs(self.v) < 0.02:
            self.t_still = t if self.t_still is None else self.t_still
        else:
            self.t_still = None
        self.ekf.add_speed(t, self.v)

    def imu_cb(self, msg):
        t = self.now()
        g = msg.angular_velocity
        raw = self.gyro_scale * float(self.up @ np.array([g.x, g.y, g.z]))
        if self.estimate_bias and self.t_still is not None and t - self.t_still > 0.5:
            self.bias += 0.01 * (raw - self.bias)  # standing: whatever the gyro shows is its bias
        self.ekf.add_yaw_rate(t, raw - self.bias)

    def pose_cb(self, msg):
        p = msg.pose.pose
        yaw = yaw_of(p.orientation)
        self.pose = (self.now() - self.pose_lag, p.position.x - self.pose_lever * math.cos(yaw),
                     p.position.y - self.pose_lever * math.sin(yaw), yaw)

    # ------------------------------------------------------------------ observations
    def clean(self, msg):
        """-> [(confidence, beam time, ego-frame point)] that pass the checks, best first"""
        t_scan, t_img = stamp_sec(msg.header.stamp), stamp_sec(msg.image_stamp)
        kept = []
        for obs in msg.observations:
            x, y, conf = float(obs.x), float(obs.y), float(obs.confidence)
            t = t_scan + float(obs.time_offset)
            r = math.hypot(x, y)
            if not (math.isfinite(r) and self.min_range <= r <= self.max_range):
                self.drops['range'] += 1
            elif conf < self.min_conf:
                self.drops['confidence'] += 1
            elif abs(math.atan2(y, x)) > self.max_bearing:
                self.drops['bearing'] += 1
            elif self.max_skew > 0.0 and t_img > 0.0 and abs(t_img - t) > self.max_skew:
                self.drops['skew'] += 1
            else:
                kept.append((conf, t, self.laser_rot @ np.array([x, y]) + self.laser_xy))
        kept.sort(key=lambda k: -k[0])
        return kept

    def observations_cb(self, msg):
        self.n_msgs += 1
        self.n_obs += len(msg.observations)
        kept = self.clean(msg)
        # One time per frame: that of the most confident point. The points of one frame are
        # alternatives for the same opponent and are a few beams apart (a few ms).
        t = kept[0][1] if kept else stamp_sec(msg.header.stamp)
        self.latency += self.now() - t
        result = self.ekf.add_detections(t, [k[2] for k in kept], self.laser_xy)
        self.counts[result] += 1
        if result == 'stale':
            self.get_logger().warning(f'observation {self.now() - t:.2f} s old (> ekf.history {self.ekf.p.history}): '
                                      'check the clocks', throttle_duration_sec=5.0)
        if self.points_pub is not None:
            out = PoseArray()
            out.header.stamp, out.header.frame_id = msg.header.stamp, self.base_frame
            for _, _, pt in kept:
                pose = Pose()
                pose.position.x, pose.position.y, pose.orientation.w = float(pt[0]), float(pt[1]), 1.0
                out.poses.append(pose)
            self.points_pub.publish(out)

    # ------------------------------------------------------------------ output
    def fill(self, x, P, frame, stamp):
        out = Odometry()
        out.header.stamp, out.header.frame_id, out.child_frame_id = stamp, frame, self.child_frame
        out.pose.pose.position.x, out.pose.pose.position.y = float(x[PX]), float(x[PY])
        out.pose.pose.orientation.z, out.pose.pose.orientation.w = math.sin(0.5 * x[PHI]), math.cos(0.5 * x[PHI])
        pc, idx = np.zeros((6, 6)), (PX, PY, PHI)
        for a, i in zip((0, 1, 5), idx):
            for b, j in zip((0, 1, 5), idx):
                pc[a, b] = P[i, j]
        out.pose.covariance = pc.ravel().tolist()
        out.twist.twist.linear.x, out.twist.twist.angular.z = float(x[VO]), float(x[WO])
        tc = np.zeros((6, 6))
        tc[0, 0], tc[0, 5], tc[5, 0], tc[5, 5] = P[VO, VO], P[VO, WO], P[WO, VO], P[WO, WO]
        out.twist.covariance = tc.ravel().tolist()
        return out

    def publish(self):
        now = self.get_clock().now()
        est = self.ekf.estimate(now.nanoseconds * 1e-9)
        pose = self.pose
        est_w = self.ekf.estimate(pose[0]) if pose is not None and self.world_pub is not None else None
        self.log(est)
        if est is None or not est.confirmed:
            return
        self.pub.publish(self.fill(est.x, est.P, self.base_frame, now.to_msg()))
        # map frame: the estimate at the time of the ego pose, so both describe the same instant
        if est_w is None or not est_w.confirmed or now.nanoseconds * 1e-9 - pose[0] > self.ekf.p.history:
            return
        _, ex, ey, eyaw = pose
        c, s = math.cos(eyaw), math.sin(eyaw)
        x = est_w.x.copy()
        x[PX] = ex + c * est_w.x[PX] - s * est_w.x[PY]
        x[PY] = ey + s * est_w.x[PX] + c * est_w.x[PY]
        x[PHI] = math.atan2(math.sin(est_w.x[PHI] + eyaw), math.cos(est_w.x[PHI] + eyaw))
        J = np.eye(len(x))
        J[PX:PY + 1, PX:PY + 1] = [[c, -s], [s, c]]
        stamp = Time(nanoseconds=int(pose[0] * 1e9)).to_msg()
        self.world_pub.publish(self.fill(x, J @ est_w.P @ J.T, self.world_frame, stamp))

    def log(self, est):
        if time.monotonic() - self.t_log < 5.0:
            return
        dt = time.monotonic() - self.t_log
        if est is not None and est.active:
            x = est.x
            state = (f'opponent{"" if est.confirmed else " (tentative)"} ({x[PX]:.2f}, {x[PY]:+.2f}) m, heading '
                     f'{math.degrees(x[PHI]):+.0f} deg, speed {x[VO]:.2f} +- {math.sqrt(est.P[VO, VO]):.2f} m/s')
        else:
            state = 'no opponent tracked'
        counts = ', '.join(f'{k} {v}' for k, v in self.counts.items() if v) or 'no observations'
        drops = ', '.join(f'{k} {v}' for k, v in self.drops.items() if v) or 'none'
        self.get_logger().info(f'{state} | {self.n_msgs / dt:.1f} Hz, {self.n_obs} points, latency '
                               f'{1e3 * self.latency / max(self.n_msgs, 1):.0f} ms | frames: {counts} | '
                               f'dropped: {drops} | ego {self.v:.2f} m/s, gyro bias {self.bias:+.4f} rad/s')
        self.counts = dict.fromkeys(RESULTS, 0)
        self.drops = dict.fromkeys(DROPS, 0)
        self.n_msgs, self.n_obs, self.latency, self.t_log = 0, 0, 0.0, time.monotonic()


def main(args=None):
    rclpy.init(args=args)
    node = OpponentEKFNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
