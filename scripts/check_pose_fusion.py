#!/usr/bin/env python3
"""Check the high-rate pose (mppi/scripts/pose_fusion.py) against a bag from the car.

    python3 scripts/check_pose_fusion.py <bag directory>

Needs /pf/pose/odom, /odom (VESC), /sensors/imu/raw and /drive in the bag. There is no ground
truth on the car, so the check is self-consistency: start from one particle-filter pose, move it
forward with the recorded motion, and compare with the particle-filter pose that came n poses
later. A good predictor lands on it. Reported:
  1. gyro scale / sign / bias (the VESC IMU reports deg/s)
  2. the particle filter's lag: the shift between pose and motion that predicts best
  3. the error of three ways to know where the car is ~0.2 s after a pose:
       hold     use the pose as it is (no compensation)
       command  speed + the steering COMMANDS through the kinematic model (what latency_comp does)
       gyro     speed + measured yaw rate (the high-rate pose)
  4. a replay through PoseFuser exactly as the node runs it: how far each new particle-filter
     pose moves the estimate
"""
import bisect
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'mppi' / 'scripts'))
from pose_fusion import PoseFuser, _advance  # noqa: E402

WB = 0.33


def load(bag):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r = rosbag2_py.SequentialReader()
    storage = 'mcap' if list(pathlib.Path(bag).glob('*.mcap')) else 'sqlite3'
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage), rosbag2_py.ConverterOptions('', ''))
    want = ('/pf/pose/odom', '/odom', '/sensors/imu/raw', '/drive')
    cls = {t.name: get_message(t.type) for t in r.get_all_topics_and_types() if t.name in want}
    out = {k: [] for k in want}
    yaw = lambda q: np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
    while r.has_next():
        topic, data, t = r.read_next()
        if topic not in cls:
            continue
        m, t = deserialize_message(data, cls[topic]), t * 1e-9
        if topic == '/pf/pose/odom':
            p = m.pose.pose
            out[topic].append((t, p.position.x, p.position.y, yaw(p.orientation)))
        elif topic == '/odom':
            out[topic].append((t, m.twist.twist.linear.x))
        elif topic == '/sensors/imu/raw':
            out[topic].append((t, m.angular_velocity.z))
        else:
            out[topic].append((t, m.drive.steering_angle))
    return {k: np.array(v) for k, v in out.items()}


class Motion:
    """speed and yaw rate over time, zero-order hold, with fast lookup"""
    def __init__(self, t, v, w):
        self.t, self.v, self.w = list(t), np.asarray(v), np.asarray(w)

    def propagate(self, pose, t0, t1, lever):
        x, y, yaw = pose
        k = max(bisect.bisect_right(self.t, t0) - 1, 0)
        t = t0
        while t < t1:
            t_next = self.t[k + 1] if k + 1 < len(self.t) else np.inf
            t_end = min(t1, max(t_next, t))
            x, y, yaw = _advance(x, y, yaw, self.v[k], self.w[k], t_end - t, lever)
            t = t_end
            if t_next <= t:
                k += 1
        return x, y, yaw


def errors(pf, motion, lag, n, lever, idx):
    """prediction of pose k+n from pose k, both taken as `lag` s old: (along, cross, yaw) errors"""
    out = []
    for k in idx:
        x, y, yaw = motion.propagate(pf[k, 1:4], pf[k, 0] - lag, pf[k + n, 0] - lag, lever)
        dx, dy = pf[k + n, 1] - x, pf[k + n, 2] - y
        c, s = np.cos(pf[k + n, 3]), np.sin(pf[k + n, 3])
        out.append((c * dx + s * dy, -s * dx + c * dy,
                    np.arctan2(np.sin(pf[k + n, 3] - yaw), np.cos(pf[k + n, 3] - yaw))))
    return np.array(out)


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def main():
    D = load(sys.argv[1])
    pf, od, im, dr = D['/pf/pose/odom'], D['/odom'], D['/sensors/imu/raw'], D['/drive']
    v_at = lambda t: np.interp(t, od[:, 0], od[:, 1])
    print(f'{len(pf)} poses ({1 / np.median(np.diff(pf[:, 0])):.1f} Hz), '
          f'{len(od)} speed and {len(im)} gyro samples ({1 / np.median(np.diff(im[:, 0])):.0f} Hz)')

    # 1. gyro: bias while standing, scale from the yaw the particle filter saw over 1 s windows
    still = v_at(im[:, 0]) < 0.02
    bias = float(np.mean(im[still, 1])) if still.sum() > 20 else 0.0
    g_int = np.r_[0.0, np.cumsum((im[1:, 1] + im[:-1, 1]) * 0.5 * np.diff(im[:, 0]))] - bias * (im[:, 0] - im[0, 0])
    yaw_pf = np.unwrap(pf[:, 3])
    tt = np.arange(pf[0, 0] + 1.0, pf[-1, 0] - 1.0, 0.25)
    dy_pf = np.interp(tt + 0.5, pf[:, 0], yaw_pf) - np.interp(tt - 0.5, pf[:, 0], yaw_pf)
    dg = np.interp(tt + 0.375, im[:, 0], g_int) - np.interp(tt - 0.625, im[:, 0], g_int)  # ~0.125 s earlier
    scale = float(np.sum(dy_pf * dg) / np.sum(dg * dg))
    print(f'1. gyro: bias {bias:+.3f} raw units ({still.sum()} samples standing), scale {scale:+.5f} rad/s per unit '
          f'(deg/s would be {np.pi / 180:.5f}), fit residual {np.degrees(rms(dy_pf - scale * dg)):.1f} deg per 1 s')
    w_gyro = scale * (im[:, 1] - bias)
    gyro = Motion(im[:, 0], v_at(im[:, 0]), w_gyro)

    moving = np.nonzero(v_at(pf[:, 0]) > 1.0)[0]
    moving = moving[moving < len(pf) - 4]

    # 2. the particle filter's lag (and which point of the car its pose is)
    print('2. lag of the particle-filter pose (prediction 3 poses ahead, ~0.19 s; rms over the moving part)')
    lags = np.arange(0.0, 0.26, 0.025)
    best = {}
    for lever in (0.0, 0.27):
        res = [errors(pf, gyro, L, 3, lever, moving[::2]) for L in lags]
        tot = [np.hypot(rms(e[:, 0]), rms(e[:, 1])) for e in res]
        j = int(np.argmin(tot))
        best[lever] = (lags[j], tot[j])
        print(f'   pose point {lever:.2f} m ahead of the rear axle: ' + ' '.join(
            f'{1e3 * L:.0f}ms:{100 * e:.1f}' for L, e in zip(lags, tot)) + ' [cm]')
        print(f'      best lag {1e3 * lags[j]:.0f} ms -> along {100 * rms(res[j][:, 0]):.1f} cm, '
              f'cross {100 * rms(res[j][:, 1]):.1f} cm, yaw {np.degrees(rms(res[j][:, 2])):.2f} deg')
    lever = min(best, key=lambda k: best[k][1])
    lag = float(best[lever][0])
    print(f'   -> lag {1e3 * lag:.0f} ms, pose point {lever:.2f} m ahead of the rear axle')

    # 3. three ways to know where the car is n poses after a pose
    steer = np.interp(im[:, 0], dr[:, 0], dr[:, 1])  # commands on the gyro time grid
    k_hold = np.clip(np.searchsorted(dr[:, 0], im[:, 0], side='right') - 1, 0, len(dr) - 1)
    steer = dr[k_hold, 1]
    v_im = v_at(im[:, 0])
    command = Motion(im[:, 0], v_im, v_im * np.tan(steer) / WB)
    zero = Motion(im[:, 0], np.zeros(len(im)), np.zeros(len(im)))
    print('3. error of the pose n poses later [along cm / cross cm / yaw deg], rms over the moving part')
    for n in (1, 2, 3):
        dt = float(np.median(pf[n:, 0] - pf[:-n, 0]))
        row = f'   {1e3 * dt:4.0f} ms ahead:'
        for name, mo in (('hold', zero), ('command', command), ('gyro', gyro)):
            e = errors(pf, mo, lag, n, lever, moving)
            row += f'  {name} {100 * rms(e[:, 0]):5.1f} / {100 * rms(e[:, 1]):4.1f} / {np.degrees(rms(e[:, 2])):4.1f}'
        print(row)

    # 4. replay through PoseFuser as the node runs it
    f = PoseFuser(lag=lag, lever=lever)
    ev = sorted([(t, 0, k) for k, t in enumerate(im[:, 0])] + [(t, 1, k) for k, t in enumerate(pf[:, 0])])
    corr, lead = [], []
    for t, kind, k in ev:
        if kind == 0:
            f.add_motion(t, v_im[k], w_gyro[k])
        else:
            had = f.anchor is not None
            f.add_pose(t, pf[k, 1], pf[k, 2], pf[k, 3])
            if had and v_at(t) > 1.0:
                corr.append(f.last_correction)
                x, y, _ = f.pose(t)
                lead.append(np.hypot(x - pf[k, 1], y - pf[k, 2]))
    corr = np.array(corr)
    print(f'4. replay: each new pose moves the estimate by {100 * rms(np.hypot(corr[:, 0], corr[:, 1])):.1f} cm rms '
          f'(p95 {100 * np.percentile(np.hypot(corr[:, 0], corr[:, 1]), 95):.1f} cm), '
          f'{np.degrees(rms(corr[:, 2])):.2f} deg rms; the estimate is {100 * np.mean(lead):.0f} cm '
          f'ahead of the raw pose on average (max {100 * np.max(lead):.0f} cm)')


if __name__ == '__main__':
    main()
