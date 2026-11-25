#!/usr/bin/env python3
"""Follow the opponent at a fixed bumper-to-bumper gap (hallway experiment, straight line).

    /opponent/odom (opponent_ekf_node) --\\
    /odom          (VESC speed)          >-- follow_node --> /drive
    /scan          (safety only)       --/                  log CSV (log_dir)
    /joy           (deadman)          --/

Speed: drive at the opponent's estimated speed, plus a correction toward the gap reference and a
damping term on the closing speed (PD on the gap):

    v_cmd = v_opp + k_gap * (gap - gap_ref) + k_rel * (v_opp - v_ego)    (k_close when too close)

The ego car matches the opponent's speed and holds gap_ref whatever that speed is. k_rel makes it
back off as soon as it closes in, before the gap has shrunk: without it, a human braking from
1.8 m/s closed a 0.5 m gap to ~0 in simulation (estimator + motor lag). v_cmd is limited to [0, max_speed], and
its change to max_accel / max_decel.

gap = opponent center x - opp_rear_to_center - ego_axle_to_front (the EKF publishes the opponent's
CENTER relative to the ego REAR AXLE). Measure both lengths on the cars.

Steering: 'pursuit' steers toward the opponent's center (pure pursuit, the opponent is the
lookahead point), so the ego stays behind it if the human drifts; 'straight' keeps 0.

Deadman: the car only drives while joy button deadman_button is held (8 = left trigger on the
8BitDo Ultimate 2C; -1 = no deadman). Released, or no /joy message for deadman_timeout (node or
controller gone): speed 0 at once, and the command ramps up from 0 again when it is pressed. The
car stack's ackermann_mux passes /drive without any button, so this is the only deadman for following.

Stops (speed 0, steering 0), whatever the filter says:
    - deadman released or /joy stale: immediately
    - no confirmed track for track_timeout (lost, or the EKF node is not running): brake at max_decel
    - gap < min_gap from the filter: immediately
    - LiDAR: any return in the corridor ahead (|y| < corridor_half_width) closer than estop_gap to the
      front bumper: immediately. Independent of the camera and the filter.

The CSV has one row per control step: time, gap (filter and raw LiDAR), speeds (ego, opponent
estimate, optional ground truth from truth_topic), the command and why. scripts/plot_follow.py
plots it.
"""
import csv
import math
import os
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy, LaserScan

FIELDS = ('t', 'state', 'gap', 'gap_lidar', 'v_ego', 'v_opp', 'v_opp_sd', 'v_opp_truth', 'opp_y', 'v_cmd',
          'steer')


def stamp_sec(stamp):
    return stamp.sec + 1e-9 * stamp.nanosec


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2))


class FollowNode(Node):
    def __init__(self):
        super().__init__('follow_node')
        defaults = {
            'opponent_topic': '/opponent/odom',
            'odom_topic': '/odom',
            'scan_topic': '/scan',
            'drive_topic': '/drive',
            'truth_topic': '',             # opponent's own odometry (its VESC), for the log only; '' = off
            'rate': 50.0,
            # gap [m], bumper to bumper
            'gap_ref': 0.5,
            'min_gap': 0.2,
            'ego_axle_to_front': 0.45,     # ego rear axle -> front bumper (measure)
            'opp_rear_to_center': 0.28,    # opponent rear bumper -> center (= ekf.surface_offset)
            # speed
            'k_gap': 1.0,                  # [1/s] gap error -> speed, when too far
            'k_close': 1.5,                # [1/s] when too close
            'k_rel': 0.6,                  # [-] closing speed -> less speed
            'max_speed': 2.0,              # [m/s] raise step by step
            'max_accel': 2.0,              # [m/s^2] on the command
            'max_decel': 4.0,
            'min_track_speed': 0.0,        # [m/s] below this opponent estimate, v_opp counts as 0
            'track_timeout': 0.3,          # [s] /opponent/odom older than this: stop
            # steering
            'steer_mode': 'pursuit',       # pursuit | straight
            'wheelbase': 0.33,
            'max_steer': 0.25,
            # LiDAR emergency stop
            'laser_x': 0.27,               # LiDAR ahead of the rear axle
            'corridor_half_width': 0.2,
            'estop_gap': 0.15,             # [m] from the front bumper
            # deadman
            'joy_topic': '/joy',
            'deadman_button': 8,           # joy button to hold (8 = left trigger, 8BitDo Ultimate 2C); -1 = off
            'deadman_timeout': 0.3,        # [s] /joy older than this counts as released
            # log
            'log_dir': os.path.expanduser('~/follow_logs'),   # '' = no CSV
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        gp = lambda k: self.get_parameter(k).value
        self.p = {k: gp(k) for k in defaults}
        p = self.p
        self.lidar_to_front = float(p['ego_axle_to_front']) - float(p['laser_x'])

        self.opp = None          # (receive time, Odometry)
        self.v_ego = 0.0
        self.v_truth = math.nan
        self.gap_lidar = math.inf
        self.deadman_button = int(p['deadman_button'])
        self.joy = None          # (receive time, deadman button held)
        self.v_cmd = 0.0
        self.t_last = None
        self.state = 'wait'
        self.n_state = {}
        self.t_log = time.monotonic()

        self.create_subscription(Odometry, p['opponent_topic'], self.opp_cb, 10)
        self.create_subscription(Odometry, p['odom_topic'], self.odom_cb, 20)
        self.create_subscription(LaserScan, p['scan_topic'], self.scan_cb, qos_profile_sensor_data)
        if p['truth_topic'] not in ('', 'none'):
            self.create_subscription(Odometry, p['truth_topic'], self.truth_cb, 20)
        if self.deadman_button >= 0:
            self.create_subscription(Joy, p['joy_topic'], self.joy_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, p['drive_topic'], 1)
        self.create_timer(1.0 / float(p['rate']), self.step)

        self.csv, self.writer = None, None
        if p['log_dir'] not in ('', 'none'):
            os.makedirs(p['log_dir'], exist_ok=True)
            path = os.path.join(p['log_dir'], time.strftime('follow_%Y%m%d_%H%M%S.csv'))
            self.csv = open(path, 'w', newline='')
            self.writer = csv.writer(self.csv)
            self.writer.writerow(FIELDS)
            self.get_logger().info(f'logging to {path}')
        self.get_logger().info(
            f'follow: gap {p["gap_ref"]} m (stop < {p["min_gap"]}), max {p["max_speed"]} m/s, steer {p["steer_mode"]}; '
            f'{p["opponent_topic"]} + {p["odom_topic"]} + {p["scan_topic"]} -> {p["drive_topic"]}')
        if self.deadman_button >= 0:
            self.get_logger().info(f'deadman: hold joy button {self.deadman_button} ({p["joy_topic"]}) to drive')
        else:
            self.get_logger().warning('deadman off (deadman_button < 0): the car drives as soon as it has a track')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ inputs
    def opp_cb(self, msg):
        self.opp = (self.now(), msg)

    def joy_cb(self, msg):
        b = self.deadman_button
        self.joy = (self.now(), b < len(msg.buttons) and msg.buttons[b] == 1)

    def deadman_held(self, t):
        if self.deadman_button < 0:
            return True
        return self.joy is not None and self.joy[1] and t - self.joy[0] < float(self.p['deadman_timeout'])

    def odom_cb(self, msg):
        self.v_ego = msg.twist.twist.linear.x

    def truth_cb(self, msg):
        self.v_truth = msg.twist.twist.linear.x

    def scan_cb(self, msg):
        r = np.asarray(msg.ranges, dtype=np.float64)
        a = msg.angle_min + msg.angle_increment * np.arange(r.size)
        ok = np.isfinite(r) & (r >= max(msg.range_min, 0.02)) & (r <= msg.range_max)
        x, y = r * np.cos(a), r * np.sin(a)
        ahead = ok & (x > 0.0) & (np.abs(y) < float(self.p['corridor_half_width']))
        self.gap_lidar = float(x[ahead].min()) - self.lidar_to_front if ahead.any() else math.inf

    # ------------------------------------------------------------------ control
    def step(self):
        p = self.p
        t = self.now()
        dt = 0.0 if self.t_last is None else min(t - self.t_last, 0.1)
        self.t_last = t
        gap = v_opp = v_sd = opp_y = math.nan
        steer, target, state = 0.0, 0.0, 'follow'

        fresh = self.opp is not None and t - self.opp[0] < float(p['track_timeout'])
        if fresh:
            m = self.opp[1]
            px, py = m.pose.pose.position.x, m.pose.pose.position.y
            phi = yaw_of(m.pose.pose.orientation)
            opp_y = py
            gap = px - float(p['opp_rear_to_center']) - float(p['ego_axle_to_front'])
            v_opp = m.twist.twist.linear.x * math.cos(phi)       # along the ego's heading
            v_sd = math.sqrt(max(m.twist.covariance[0], 0.0))
            if v_opp < float(p['min_track_speed']):
                v_opp = 0.0
            err = gap - float(p['gap_ref'])
            k = float(p['k_gap']) if err >= 0.0 else float(p['k_close'])
            target = v_opp + k * err + float(p['k_rel']) * (v_opp - self.v_ego)
            if p['steer_mode'] == 'pursuit':
                d2 = max(px * px + py * py, 0.25)
                steer = math.atan(2.0 * float(p['wheelbase']) * py / d2)

        # stops, most urgent first
        hard = False
        if not self.deadman_held(t):
            state, hard = 'deadman', True
        elif self.gap_lidar < float(p['estop_gap']):
            state, hard = 'estop_lidar', True
        elif fresh and gap < float(p['min_gap']):
            state, hard = 'too_close', True
        elif not fresh:
            state = 'no_track' if self.opp is not None else 'wait'
            target = 0.0
            steer = 0.0

        target = float(np.clip(target, 0.0, float(p['max_speed'])))
        if hard:
            self.v_cmd, steer = 0.0, 0.0
        else:
            dv = target - self.v_cmd
            self.v_cmd += float(np.clip(dv, -float(p['max_decel']) * dt, float(p['max_accel']) * dt))
        steer = float(np.clip(steer, -float(p['max_steer']), float(p['max_steer'])))

        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        out.drive.speed = self.v_cmd
        out.drive.steering_angle = steer
        self.drive_pub.publish(out)

        if self.writer is not None:
            self.writer.writerow([f'{t:.4f}', state, f'{gap:.4f}', f'{self.gap_lidar:.4f}', f'{self.v_ego:.4f}',
                                  f'{v_opp:.4f}', f'{v_sd:.4f}', f'{self.v_truth:.4f}', f'{opp_y:.4f}',
                                  f'{self.v_cmd:.4f}', f'{steer:.4f}'])
        self.n_state[state] = self.n_state.get(state, 0) + 1
        if state != self.state and state != 'follow':
            self.get_logger().warning(f'{state}: stopping (gap {gap:.2f} m, lidar gap {self.gap_lidar:.2f} m)')
        self.state = state
        if time.monotonic() - self.t_log > 2.0:
            self.t_log = time.monotonic()
            states = ', '.join(f'{k} {v}' for k, v in self.n_state.items())
            self.get_logger().info(f'gap {gap:.2f} m (lidar {self.gap_lidar:.2f}), opponent {v_opp:.2f} +- {v_sd:.2f} '
                                   f'm/s, ego {self.v_ego:.2f}, cmd {self.v_cmd:.2f} m/s | {states}')
            self.n_state = {}

    def stop(self):
        out = AckermannDriveStamped()
        self.drive_pub.publish(out)
        if self.csv is not None:
            self.csv.close()


def main(args=None):
    rclpy.init(args=args)
    node = FollowNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.stop()
        except Exception:
            pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
