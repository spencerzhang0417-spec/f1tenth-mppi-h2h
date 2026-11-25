#!/usr/bin/env python3
"""Rates and delays of the perception -> EKF chain, measured at a subscriber (needs the workspace sourced).

    python3 measure_latency.py --duration 60 --out resources/perception_ekf_latency.json

observation age  = receive time - LaserScan stamp of the observation (camera frame + inference + DDS)
image - scan     = camera frame-retrieval time - scan stamp (how old the scan was when the image was taken)
scan age         = receive time - scan stamp (LiDAR driver + DDS only)
"""
import argparse, json, time

import numpy as np
import rclpy
from car_observation_msgs.msg import CarObservationArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

st = lambda s: s.sec + 1e-9 * s.nanosec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--duration', type=float, default=60.0); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rclpy.init(); n = Node('measure_latency'); now = lambda: n.get_clock().now().nanoseconds * 1e-9
    d = {'obs_t': [], 'obs_age': [], 'img_minus_scan': [], 'obs_n': [], 'scan_t': [], 'scan_age': [], 'opp_t': [], 'opp_age': []}
    def obs(m):
        t = now(); d['obs_t'].append(t); d['obs_age'].append(t - st(m.header.stamp)); d['img_minus_scan'].append(st(m.image_stamp) - st(m.header.stamp)); d['obs_n'].append(len(m.observations))
    def scan(m):
        t = now(); d['scan_t'].append(t); d['scan_age'].append(t - st(m.header.stamp))
    def opp(m):
        t = now(); d['opp_t'].append(t); d['opp_age'].append(t - st(m.header.stamp))
    n.create_subscription(CarObservationArray, '/perception/car_observations', obs, 10)
    n.create_subscription(LaserScan, '/scan', scan, qos_profile_sensor_data)
    n.create_subscription(Odometry, '/opponent/odom', opp, 10)
    t_end = time.time() + a.duration
    while time.time() < t_end:
        rclpy.spin_once(n, timeout_sec=0.05)
    def s(v, scale=1000.0):
        v = np.asarray(v, float) * scale
        return None if not len(v) else {'mean': round(float(v.mean()), 1), 'p50': round(float(np.percentile(v, 50)), 1), 'p95': round(float(np.percentile(v, 95)), 1), 'max': round(float(v.max()), 1)}
    def rate(t):
        return None if len(t) < 2 else round((len(t) - 1) / (t[-1] - t[0]), 2)
    def gaps(t):
        return None if len(t) < 2 else s(np.diff(t))
    out = {'duration_s': a.duration,
           'observations': {'count': len(d['obs_t']), 'rate_hz': rate(d['obs_t']), 'period_ms': gaps(d['obs_t']), 'age_ms': s(d['obs_age']),
                            'image_minus_scan_ms': s(d['img_minus_scan']), 'frames_with_detection_pct': round(100 * float(np.mean(np.array(d['obs_n']) > 0)), 1) if d['obs_n'] else None},
           'scan': {'count': len(d['scan_t']), 'rate_hz': rate(d['scan_t']), 'age_ms': s(d['scan_age'])},
           'opponent_odom': {'count': len(d['opp_t']), 'rate_hz': rate(d['opp_t']), 'period_ms': gaps(d['opp_t']), 'stamp_age_ms': s(d['opp_age'])}}
    json.dump(out, open(a.out, 'w'), indent=2); print(json.dumps(out, indent=2))
    n.destroy_node(); rclpy.try_shutdown()


if __name__ == '__main__':
    main()
