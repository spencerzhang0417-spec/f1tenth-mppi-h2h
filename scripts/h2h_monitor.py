#!/usr/bin/env python3
"""Watch a head-to-head run in the simulator and print its numbers.

    python3 scripts/h2h_monitor.py --duration 60 --opps 1 [--out run.json]

Start it next to `ros2 launch mppi h2h_sim_launch.py`. For the ego and every opponent it
follows the progress along the raceline (unwrapped, so laps add up) and reports:
  - ego lap times, mean / max speed
  - overtakes: the ego goes from behind an opponent to >= half a car length ahead of it
    (the rule-book definition) and the time it spent within 2 m behind before that
  - collisions: rising edges of the simulator's collision flags
  - closest distance between the ego and each opponent (center to center)
"""
import argparse
import json
import pathlib
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool

CAR_LENGTH = 0.58


class Line:
    def __init__(self, csv):
        xy = np.loadtxt(csv, delimiter=',', comments='#', ndmin=2)[:, :2]
        seg = np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T)
        self.xy, self.s, self.length = xy, np.r_[0.0, np.cumsum(seg)[:-1]], float(seg.sum())

    def progress(self, x, y):
        return float(self.s[int(np.argmin(np.hypot(self.xy[:, 0] - x, self.xy[:, 1] - y)))])


class Car:
    def __init__(self, name):
        self.name, self.xy, self.v, self.stamp = name, None, 0.0, 0.0
        self.s_prev, self.s_total, self.s_start = None, 0.0, None
        self.collisions, self.in_collision = 0, False
        self.lap_marks, self.v_sum, self.v_max, self.n = [], 0.0, 0.0, 0

    def update(self, line, x, y, v, t):
        s = line.progress(x, y)
        if self.s_prev is None:
            self.s_start = s
        else:
            d = s - self.s_prev
            d += line.length if d < -0.5 * line.length else (-line.length if d > 0.5 * line.length else 0.0)
            before = int((self.s_start + self.s_total) // line.length)
            self.s_total += d
            if int((self.s_start + self.s_total) // line.length) > before:
                self.lap_marks.append(t)
        self.s_prev, self.xy, self.v = s, (x, y), v
        self.v_sum += v
        self.v_max = max(self.v_max, v)
        self.n += 1


class Monitor(Node):
    def __init__(self, line, n_opps):
        super().__init__('h2h_monitor')
        self.line, self.t0 = line, None
        self.ego = Car('ego_racecar')
        self.opps = [Car('opp_racecar' + ('' if i == 0 else str(i + 1))) for i in range(n_opps)]
        self.gap0 = [None] * n_opps      # ego progress minus opponent progress at the start [m]
        self.behind = [0.0] * n_opps     # time spent within 2 m behind [s]
        self.passed = [None] * n_opps    # time of the overtake [s]
        self.repassed = [0] * n_opps     # the opponent took the place back
        self.ahead = [False] * n_opps
        self.min_dist = [np.inf] * n_opps
        self.t_last = None
        self.contacts = []
        for car in [self.ego] + self.opps:
            self.create_subscription(Odometry, f'/{car.name}/odom', lambda m, c=car: self.odom(m, c), 10)
            self.create_subscription(Bool, f'/{car.name}/collision', lambda m, c=car: self.coll(m, c), 10)

    def now(self):
        t = time.monotonic()
        if self.t0 is None:
            self.t0 = t
        return t - self.t0

    def coll(self, msg, car):
        if msg.data and not car.in_collision:
            car.collisions += 1
            if car.xy is not None:  # when and where: the stamp is the car's own odometry stamp
                other = [o for o in [self.ego] + self.opps if o is not car and o.xy is not None]
                near = min((float(np.hypot(o.xy[0] - car.xy[0], o.xy[1] - car.xy[1])) for o in other), default=None)
                self.contacts.append({'car': car.name, 't_s': round(self.now(), 2), 'stamp': round(car.stamp, 3),
                                      'x': round(car.xy[0], 2), 'y': round(car.xy[1], 2),
                                      's_m': round(self.line.progress(*car.xy), 1), 'v': round(car.v, 2),
                                      'nearest_car_m': None if near is None else round(near, 2)})
        car.in_collision = bool(msg.data)

    def odom(self, msg, car):
        t = self.now()
        p = msg.pose.pose.position
        car.update(self.line, p.x, p.y, msg.twist.twist.linear.x, t)
        car.stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if car is not self.ego or self.ego.xy is None:
            return
        dt = 0.0 if self.t_last is None else t - self.t_last
        self.t_last = t
        for i, opp in enumerate(self.opps):
            if opp.xy is None:
                continue
            if self.gap0[i] is None:
                g = self.ego.s_start - opp.s_start
                g -= self.line.length * round(g / self.line.length)  # nearest: -L/2 .. L/2
                self.gap0[i] = g
            gap = self.gap0[i] + self.ego.s_total - opp.s_total
            self.min_dist[i] = min(self.min_dist[i], float(np.hypot(self.ego.xy[0] - opp.xy[0],
                                                                    self.ego.xy[1] - opp.xy[1])))
            if -2.0 < gap < 0.0:
                self.behind[i] += dt
            if not self.ahead[i] and gap > 0.5 * CAR_LENGTH:
                self.ahead[i] = True
                if self.passed[i] is None:
                    self.passed[i] = t
            elif self.ahead[i] and gap < -0.5 * CAR_LENGTH:
                self.ahead[i] = False
                self.repassed[i] += 1

    def summary(self):
        e = self.ego
        laps = np.diff(e.lap_marks).round(2).tolist()
        out = {
            'duration_s': round(self.now(), 1),
            'ego': {'distance_m': round(e.s_total, 1), 'laps_completed': len(e.lap_marks),
                    'lap_times_s': laps, 'v_mean': round(e.v_sum / max(e.n, 1), 2),
                    'v_max': round(e.v_max, 2), 'collisions': e.collisions},
            'opponents': [],
            'contacts': self.contacts,
        }
        for i, o in enumerate(self.opps):
            gap = None if self.gap0[i] is None else round(self.gap0[i] + e.s_total - o.s_total, 2)
            out['opponents'].append({
                'name': o.name, 'distance_m': round(o.s_total, 1),
                'v_mean': round(o.v_sum / max(o.n, 1), 2), 'collisions': o.collisions,
                'start_gap_m': None if self.gap0[i] is None else round(self.gap0[i], 2),
                'final_gap_m': gap,
                'overtaken_at_s': None if self.passed[i] is None else round(self.passed[i], 1),
                'time_within_2m_behind_s': round(self.behind[i], 1),
                'lost_place_again': self.repassed[i],
                'min_center_distance_m': None if not np.isfinite(self.min_dist[i]) else round(self.min_dist[i], 2),
            })
        return out


def main():
    here = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--duration', type=float, default=60.0, help='[s] measured from the first odom message')
    ap.add_argument('--opps', type=int, default=1)
    ap.add_argument('--raceline', default=str(here / 'mppi' / 'waypoints' / 'race2_raceline.csv'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rclpy.init()
    node = Monitor(Line(args.raceline), args.opps)
    try:
        while rclpy.ok() and (node.t0 is None or node.now() < args.duration):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    out = node.summary()
    print(json.dumps(out, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
