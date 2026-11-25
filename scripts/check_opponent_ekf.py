#!/usr/bin/env python3
"""Offline checks of the opponent EKF (mppi/scripts/opponent_ekf.py). No ROS, no car.

    python3 scripts/check_opponent_ekf.py                  # all checks, prints the EKF errors
    python3 -m pytest scripts/check_opponent_ekf.py -q     # the same as tests

Two cars drive curvy paths at racing pace; the filter gets, in ARRIVAL order, the gyro at 200 Hz,
the VESC speed at 50 Hz (both on time) and lab 10 observations at 20 Hz (one LiDAR surface point
per detected car) that describe the beam time but arrive 80 ms later. The observations are noisy
the way a single LiDAR beam picked by a camera box is: 20 % of the frames have no car, 10 % of
the points slipped past the car onto the background (range too long), 10 % of the frames carry an
extra false positive. The error is scored on the estimate predicted to the arrival time ("now"),
as the node publishes it.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'mppi' / 'scripts'))
from opponent_ekf import EKFParams, OpponentEKF, PHI, PX, PY, VO, WO, wrap  # noqa: E402

LIDAR = np.array([0.27, 0.0])   # LiDAR ahead of the rear axle
SURFACE = 0.28                  # seen surface (rear box) -> car center


# ---------------------------------------------------------------------------- EKF
def trajectories(T=20.0, dt=0.001):
    t = np.arange(0.0, T, dt)
    ve, we = 2.5 + 1.0 * np.sin(1.3 * t), 1.0 * np.sin(1.0 * t)
    vo, wo = 2.5 + 1.2 * np.sin(1.5 * t + 1.0), 1.0 * np.sin(1.1 * t - 0.3)

    def integrate(x0, y0, v, w):
        th = np.cumsum(w) * dt
        return x0 + np.cumsum(v * np.cos(th)) * dt, y0 + np.cumsum(v * np.sin(th)) * dt, th
    return t, integrate(0.0, 0.0, ve, we), integrate(2.5, 0.3, vo, wo), (ve, we, vo, wo)


def relative(i, ego, opp):
    dx, dy = opp[0][i] - ego[0][i], opp[1][i] - ego[1][i]
    c, s = math.cos(ego[2][i]), math.sin(ego[2][i])
    return c * dx + s * dy, -s * dx + c * dy, wrap(opp[2][i] - ego[2][i])


def run(delay=0.08, seed=0, false_positive_rate=0.1, miss_rate=0.2, background_rate=0.1, noise=1.0,
        params=None, rate=20.0):
    """-> (N, 5) errors [x, y, heading, speed, yaw rate], {detection result: count}"""
    rng = np.random.default_rng(seed)
    t, ego, opp, (ve, we, vo, wo) = trajectories()
    dt = t[1] - t[0]
    ekf = OpponentEKF(params or EKFParams(surface_offset=SURFACE))
    arrivals = []  # (arrival time, order, kind, measurement time, data)
    for k in range(0, len(t), int(round(0.005 / dt))):
        arrivals.append((t[k], 0, 'w', t[k], we[k] + noise * rng.normal(0, 0.02)))
    for k in range(0, len(t), int(round(0.02 / dt))):
        arrivals.append((t[k], 1, 'v', t[k], ve[k] + noise * rng.normal(0, 0.1)))
    for k in range(0, len(t), int(round(1 / rate / dt))):
        if rng.random() < miss_rate:
            arrivals.append((t[k] + delay, 2, 'det', t[k], []))  # empty array: no car in this frame
            continue
        px, py, _ = relative(k, ego, opp)
        d = np.array([px, py]) - LIDAR
        r = np.hypot(*d) - SURFACE + noise * rng.normal(0, 0.03)
        b = math.atan2(d[1], d[0]) + noise * rng.normal(0, 0.01)
        if rng.random() < background_rate:  # the beam missed the car: the wall behind it
            r += rng.uniform(0.5, 3.0)
        pts = [LIDAR + r * np.array([math.cos(b), math.sin(b)])]
        if rng.random() < false_positive_rate:
            pts.append(LIDAR + rng.uniform(1, 4) * np.array([1.0, rng.uniform(-1, 1)]))
            rng.shuffle(pts)
        arrivals.append((t[k] + delay, 2, 'det', t[k], pts))
    arrivals.sort(key=lambda a: (a[0], a[1]))

    errs, results = [], {}
    for t_arr, _, kind, t_meas, data in arrivals:
        if kind == 'w':
            ekf.add_yaw_rate(t_meas, data)
        elif kind == 'v':
            ekf.add_speed(t_meas, data)
        else:
            res = ekf.add_detections(t_meas, data, LIDAR)
            results[res] = results.get(res, 0) + 1
        if kind == 'v' and t_arr > 5.0:
            est = ekf.estimate(t_arr)
            k = int(round(t_arr / dt))
            if est.confirmed and k < len(t):
                px, py, phi = relative(k, ego, opp)
                x = est.x
                errs.append((x[PX] - px, x[PY] - py, wrap(x[PHI] - phi), x[VO] - vo[k], x[WO] - wo[k]))
    return np.array(errs), results


def test_tracks_opponent_with_delayed_detections():
    rms_all = []
    for seed in range(3):
        errs, results = run(seed=seed)
        assert len(errs) > 500 and results.get('reinit', 0) == 0, results
        rms_all.append(np.sqrt(np.mean(errs ** 2, axis=0)))
    rms = np.mean(rms_all, axis=0)
    print(f'EKF rms over 3 runs: x {rms[0]:.3f} m, y {rms[1]:.3f} m, heading {math.degrees(rms[2]):.1f} deg, '
          f'speed {rms[3]:.3f} m/s, yaw rate {rms[4]:.3f} rad/s')
    assert rms[0] < 0.08 and rms[1] < 0.08
    assert math.degrees(rms[2]) < 10.0
    assert rms[3] < 0.4 and rms[4] < 0.45


def test_model_is_right_without_noise():
    # what is left is the lag of a constant-speed model behind 1.8 m/s^2 maneuvers (~1 cm, ~3 deg);
    # a wrong sign or Jacobian gives decimeters
    errs, _ = run(delay=0.0, false_positive_rate=0.0, miss_rate=0.0, background_rate=0.0, noise=0.0, rate=30.0)
    rms = np.sqrt(np.mean(errs ** 2, axis=0))
    assert rms[0] < 0.02 and rms[1] < 0.02 and math.degrees(rms[2]) < 5.0, rms


def test_late_detection_equals_on_time_detection():
    a, b = OpponentEKF(), OpponentEKF()
    for k in range(100):
        t = 0.01 * k
        for f in (a, b):
            f.add_speed(t, 1.0)
            f.add_yaw_rate(t, 0.1)
        if k == 50:
            a.add_detections(t - 0.07, [(3.0, 0.5)], LIDAR)   # arrives 70 ms late
        if k == 43:
            b.add_detections(t, [(3.0, 0.5)], LIDAR)          # on time
    ea, eb = a.estimate(1.0), b.estimate(1.0)
    assert np.allclose(ea.x, eb.x) and np.allclose(ea.P, eb.P)


def test_track_is_dropped_without_detections():
    f = OpponentEKF(EKFParams(lost_timeout=0.5, confirm_hits=0))
    f.add_detections(0.0, [(3.0, 0.0)])
    assert f.estimate(0.4).active and not f.estimate(0.6).active
    assert f.add_detections(0.7, [(3.0, 0.0)]) == 'init'


def test_lone_false_positive_is_never_confirmed():
    f = OpponentEKF()
    assert f.add_detections(0.0, [(2.0, 1.0)]) == 'init'
    est = f.estimate(0.1)
    assert est.active and not est.confirmed
    for k in range(1, 20):  # nothing seen afterwards
        f.add_detections(0.05 * k, [])
    assert not f.estimate(1.0).active


def test_track_is_confirmed_after_consistent_hits():
    f = OpponentEKF(EKFParams(confirm_hits=3))
    for k in range(4):
        f.add_speed(0.05 * k, 0.0)
        f.add_detections(0.05 * k, [(2.0, 0.0)])
    assert f.estimate(0.2).confirmed


def test_background_hits_do_not_pull_the_range():
    # a confirmed track at 2 m, then every other frame the beam hits a wall 1.5 m behind it
    f = OpponentEKF()
    for k in range(60):
        t = 0.05 * k
        f.add_speed(t, 0.0)
        f.add_yaw_rate(t, 0.0)
        x = 2.0 + (1.5 if k > 10 and k % 2 else 0.0)
        res = f.add_detections(t, [(x, 0.0)], LIDAR)
        if k > 10 and k % 2:
            assert res == 'reject', (k, res)
    est = f.estimate(3.0)
    assert est.confirmed and abs(est.x[PX] - (2.0 + SURFACE)) < 0.05 and abs(est.x[VO]) < 0.2, est.x


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'ok  {name}')
