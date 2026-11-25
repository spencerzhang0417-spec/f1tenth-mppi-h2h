#!/usr/bin/env python3
"""Replay a recorded run through the opponent EKF. numpy only: no ROS, no car.

    python3 replay_ekf_npz.py ../npz/opp_1125_142916.npz                     # launch parameters, motion:=straight
    python3 replay_ekf_npz.py ../npz/opp_1125_142916.npz --motion race
    python3 replay_ekf_npz.py ../npz/opp_static_1125_1337.npz --set opp_yaw_accel_std=0 init_yaw_rate_std=0.001
    python3 replay_ekf_npz.py ../npz/opp_1125_142916.npz --max-bearing 0.2   # drop detections outside +-0.2 rad

Feeds the VESC speed, the gyro and the lab 10 detections to mppi/scripts/opponent_ekf.py in the order
and at the times they arrived on the car (as opponent_ekf_node.py does), and compares the result with
  - the opponent position measured by the detections themselves (surface point + offsets), and its
    speed over ground (central difference over 6 frames, ~0.4 s, plus the ego's VESC speed): the reference
  - what the EKF published on the car that day (opp_* in the file)
replay(...) returns the estimate as arrays, for use from other scripts / the simulator.
"""
import argparse, math, pathlib, sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'mppi' / 'scripts'))
from opponent_ekf import EKFParams, OpponentEKF, PHI, PX, PY, VO, WO  # noqa: E402

LASER = np.array([0.27, 0.0])          # LiDAR ahead of the rear axle
# ekf.* of mppi/launch/opponent_ekf_launch.py (2026-11-25)
BASE = {'surface_offset': 0.28, 'opp_accel_std': 2.0, 'opp_yaw_accel_std': 1.5, 'speed_std': 0.15, 'yaw_rate_std': 0.03,
        'range_std': 0.05, 'bearing_std': 0.03, 'gate': 11.8, 'confirm_hits': 3, 'confirm_window': 0.5,
        'reinit_misses': 5, 'lost_timeout': 1.0, 'history': 1.0}
MOTION = {'race': {}, 'straight': {'opp_heading_std': 0.01, 'opp_yaw_accel_std': 0.05, 'init_yaw_rate_std': 0.1,
                                   'init_heading_std': 0.1, 'opp_accel_std': 1.5}}
# what the 2026-11-25 sweep suggests for motion:=straight (not applied on the car)
SUGGESTED = {'opp_heading_std': 0.02, 'opp_yaw_accel_std': 0.0, 'init_yaw_rate_std': 0.001, 'range_std': 0.02}


def events(d, min_conf=0.5, min_range=0.2, max_range=8.0, max_bearing=0.7, max_skew=0.1, drop=None):
    """-> [(arrival time, kind, data)] sorted by arrival. drop(t, x, y) -> True removes a detection (occlusion tests)."""
    ev = [(t, 'v', v) for t, v in zip(d['odom_t'], d['odom_v'])] + [(t, 'w', math.radians(g)) for t, g in zip(d['imu_t'], d['imu_gz'])]
    first = np.searchsorted(d['det_frame'], np.arange(len(d['obs_t'])))
    for i, t in enumerate(d['obs_t']):
        kept = []
        for j in range(first[i], first[i] + int(d['obs_n'][i])):
            x, y, c = float(d['det_x'][j]), float(d['det_y'][j]), float(d['det_conf'][j])
            tb = float(d['obs_stamp'][i]) + float(d['det_time_offset'][j]); r = math.hypot(x, y)
            if not (min_range <= r <= max_range) or c < min_conf or abs(math.atan2(y, x)) > max_bearing: continue
            if max_skew > 0 and abs(float(d['obs_image_stamp'][i]) - tb) > max_skew: continue
            if drop is not None and drop(tb, x, y): continue
            kept.append((c, tb, np.array([x, y]) + LASER))
        kept.sort(key=lambda k: -k[0])
        ev.append((t, 'o', (kept[0][1] if kept else float(d['obs_stamp'][i]), [k[2] for k in kept])))
    ev.sort(key=lambda e: e[0])
    return ev


def replay(d, params, rate=50.0, **clean):
    """-> dict of arrays t, x, y, yaw, v, w, v_std, v_fwd (speed along the ego heading, what follow_node uses)"""
    ekf = OpponentEKF(EKFParams(**params))
    t_still, bias, t_pub, out, counts = None, 0.0, None, [], {}
    for t, kind, data in events(d, **clean):
        if kind == 'v':
            t_still = (t if t_still is None else t_still) if abs(data) < 0.02 else None
            ekf.add_speed(t, data)
        elif kind == 'w':
            if t_still is not None and t - t_still > 0.5: bias += 0.01 * (data - bias)      # gyro bias while standing
            ekf.add_yaw_rate(t, data - bias)
        else:
            res = ekf.add_detections(data[0], data[1], LASER); counts[res] = counts.get(res, 0) + 1
        if t_pub is None or t - t_pub >= 1.0 / rate:
            t_pub = t; e = ekf.estimate(t)
            if e is not None and e.confirmed:
                out.append((t, e.x[PX], e.x[PY], e.x[PHI], e.x[VO], e.x[WO], math.sqrt(max(e.P[VO, VO], 0.0))))
    a = np.array(out).reshape(-1, 7)
    r = dict(zip(('t', 'x', 'y', 'yaw', 'v', 'w', 'v_std'), a.T)); r['v_fwd'] = r['v'] * np.cos(r['yaw']); r['frames'] = counts
    return r


def reference(d, surface_offset=0.28, half=3):
    """opponent center x and speed along the ego axis from the most confident detection of each frame"""
    first = np.searchsorted(d['det_frame'], np.arange(len(d['obs_t'])))
    t, x = [], []
    for i in range(len(d['obs_t'])):
        n = int(d['obs_n'][i])
        if n:
            j = first[i] + int(np.argmax(d['det_conf'][first[i]:first[i] + n]))
            t.append(float(d['obs_t'][i])); x.append(float(d['det_x'][j]) + LASER[0] + surface_offset)
    t, x = np.array(t), np.array(x); v = np.full(len(t), np.nan)
    for i in range(half, len(t) - half): v[i] = (x[i + half] - x[i - half]) / (t[i + half] - t[i - half])
    return t, x, v + np.interp(t, d['odom_t'], d['odom_v'])     # relative -> over ground (straight line)


def score(name, e, ref, ego_v=None):
    tr, xr, vr = ref
    if not len(e['t']):
        print(f"{name:28s} no confirmed track"); return
    xm, vm = np.interp(e['t'], tr, xr), np.interp(e['t'], tr, np.nan_to_num(vr))
    mov, still = np.abs(vm) > 0.3, np.abs(vm) < 0.05
    f = lambda s, a: float(np.sqrt(np.mean(a[s] ** 2))) if s.sum() > 5 else float('nan')
    dt = np.diff(e['t']); tracked = float(dt[dt < 0.2].sum())
    print(f"{name:28s} {tracked:5.1f} s tracked | position rms {f(np.ones(len(xm), bool), e['x']-xm):.3f} m | speed along ego: rms error moving {f(mov, e['v_fwd']-vm):.2f} m/s, "
          f"standing rms {f(still, e['v_fwd']):.3f} max {np.abs(e['v_fwd'][still]).max() if still.any() else float('nan'):.2f} m/s | |heading| max {np.degrees(np.abs(e['yaw'])).max():3.0f} deg | reported v std {e['v_std'].mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz'); ap.add_argument('--motion', default='straight', choices=list(MOTION))
    ap.add_argument('--set', nargs='*', default=[], help='ekf parameter overrides, name=value')
    ap.add_argument('--max-bearing', type=float, default=0.7)
    a = ap.parse_args()
    d = np.load(a.npz)
    ref = reference(d)
    over = {k: type(getattr(EKFParams(), k))(float(v)) for k, v in (s.split('=') for s in a.set)}
    car = {'t': d['opp_t'], 'x': d['opp_x'], 'yaw': d['opp_yaw'], 'v_fwd': d['opp_v'] * np.cos(d['opp_yaw']), 'v_std': d['opp_v_std']}
    print(f"{str(d['bag'])}: {len(d['obs_t'])} frames, {int((d['obs_n'] > 0).sum())} with a car, opponent x {np.nanmin(ref[1]):.2f}..{np.nanmax(ref[1]):.2f} m, |v| max {np.nanmax(np.abs(ref[2])):.2f} m/s")
    score('published on the car', car, ref)
    score(f'replay motion:={a.motion}', replay(d, {**BASE, **MOTION[a.motion]}, max_bearing=a.max_bearing), ref)
    score('replay + suggested', replay(d, {**BASE, **MOTION['straight'], **SUGGESTED}, max_bearing=a.max_bearing), ref)
    if over:
        score('replay + --set', replay(d, {**BASE, **MOTION[a.motion], **over}, max_bearing=a.max_bearing), ref)


if __name__ == '__main__':
    main()
