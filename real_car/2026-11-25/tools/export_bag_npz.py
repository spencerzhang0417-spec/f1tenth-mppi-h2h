#!/usr/bin/env python3
"""rosbag2 (mcap) -> one compressed .npz that needs only numpy to read (for tests away from the car).

    python3 export_bag_npz.py ../bags/opp_1125_142916 ../npz/         # needs ROS 2 + the workspace sourced

All times are seconds since the first message of the bag (t0, absolute, is stored too). *_t is the
time the recorder received the message; *_stamp the header stamp.

    obs_*      /perception/car_observations (lab 10): one row per FRAME
               obs_t, obs_stamp (LaserScan stamp), obs_image_stamp, obs_n (detections in the frame)
    det_*      one row per DETECTION: det_frame (row in obs_*), det_x, det_y (LiDAR frame, surface
               point), det_conf, det_beam, det_time_offset
    odom_*     /odom: odom_t, odom_v (VESC speed)
    imu_*      /sensors/imu/raw: imu_t, imu_gz (z rate as published: deg/s on this VESC)
    core_*     /sensors/core: core_t, core_voltage
    opp_*      /opponent/odom, what the EKF published on the car: opp_t, opp_x, opp_y (opponent center,
               ego rear-axle frame), opp_yaw, opp_v (along opp_yaw), opp_w, opp_v_std
    scan_*     /scan: scan_t, scan_stamp, scan_ranges [n, beams] float32, scan_angle_min,
               scan_angle_increment, scan_time_increment
    drive_*    /drive if recorded: drive_t, drive_speed, drive_steer
"""
import math, os, sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

st = lambda s: s.sec + 1e-9 * s.nanosec


def export(bag, out_dir):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: get_message(t.type) for t in r.get_all_topics_and_types()}
    want = {'/perception/car_observations', '/odom', '/sensors/imu/raw', '/sensors/core', '/opponent/odom', '/scan', '/drive'}
    M = {k: [] for k in want}
    t0 = None
    while r.has_next():
        topic, data, t = r.read_next()
        t *= 1e-9
        t0 = t if t0 is None else t0
        if topic in want:
            M[topic].append((t, deserialize_message(data, types[topic])))
    rel = lambda t: t - t0
    d = {'t0': t0, 'bag': os.path.basename(os.path.normpath(bag))}
    ob = M['/perception/car_observations']
    d['obs_t'] = [rel(t) for t, _ in ob]; d['obs_stamp'] = [rel(st(m.header.stamp)) for _, m in ob]
    d['obs_image_stamp'] = [rel(st(m.image_stamp)) for _, m in ob]; d['obs_n'] = [len(m.observations) for _, m in ob]
    det = [(i, o.x, o.y, o.confidence, o.beam_index, o.time_offset) for i, (_, m) in enumerate(ob) for o in m.observations]
    for j, k in enumerate(('det_frame', 'det_x', 'det_y', 'det_conf', 'det_beam', 'det_time_offset')):
        d[k] = np.array([x[j] for x in det], dtype=np.int32 if k in ('det_frame', 'det_beam') else np.float64)
    d['odom_t'] = [rel(t) for t, _ in M['/odom']]; d['odom_v'] = [m.twist.twist.linear.x for _, m in M['/odom']]
    d['imu_t'] = [rel(t) for t, _ in M['/sensors/imu/raw']]; d['imu_gz'] = [m.angular_velocity.z for _, m in M['/sensors/imu/raw']]
    d['core_t'] = [rel(t) for t, _ in M['/sensors/core']]; d['core_voltage'] = [m.state.voltage_input for _, m in M['/sensors/core']]
    yaw = lambda q: math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    op = M['/opponent/odom']
    d['opp_t'] = [rel(t) for t, _ in op]; d['opp_x'] = [m.pose.pose.position.x for _, m in op]; d['opp_y'] = [m.pose.pose.position.y for _, m in op]
    d['opp_yaw'] = [yaw(m.pose.pose.orientation) for _, m in op]; d['opp_v'] = [m.twist.twist.linear.x for _, m in op]
    d['opp_w'] = [m.twist.twist.angular.z for _, m in op]; d['opp_v_std'] = [math.sqrt(max(m.twist.covariance[0], 0.0)) for _, m in op]
    sc = M['/scan']
    d['scan_t'] = [rel(t) for t, _ in sc]; d['scan_stamp'] = [rel(st(m.header.stamp)) for _, m in sc]
    if sc:
        rr = np.array([np.asarray(m.ranges, dtype=np.float32) for _, m in sc]); rr[~np.isfinite(rr)] = 0.0
        d['scan_ranges'] = rr
        d['scan_angle_min'], d['scan_angle_increment'], d['scan_time_increment'] = sc[0][1].angle_min, sc[0][1].angle_increment, sc[0][1].time_increment
    d['drive_t'] = [rel(t) for t, _ in M['/drive']]; d['drive_speed'] = [m.drive.speed for _, m in M['/drive']]
    d['drive_steer'] = [m.drive.steering_angle for _, m in M['/drive']]
    out = os.path.join(out_dir, d['bag'] + '.npz')
    np.savez_compressed(out, **{k: (v if isinstance(v, np.ndarray) else np.asarray(v)) for k, v in d.items()})
    print(f"{d['bag']}: {len(ob)} observation frames ({sum(1 for n in d['obs_n'] if n)} with a car), {len(op)} EKF outputs, {len(sc)} scans, "
          f"{len(M['/drive'])} drive commands, {rel(max(v[-1][0] for v in M.values() if v)):.0f} s -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == '__main__':
    out_dir = sys.argv[-1]
    os.makedirs(out_dir, exist_ok=True)
    for bag in sys.argv[1:-1]:
        export(bag, out_dir)
