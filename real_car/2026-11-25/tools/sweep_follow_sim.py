#!/usr/bin/env python3
"""EKF / follow parameter sets on (1) the standing-cars recording and (2) a closed-loop straight-line
follow simulation fed with the real detection noise. numpy only.

    python3 sweep_follow_sim.py ../npz/opp_static_1125_1337.npz

(1) Replays the recording through the EKF and computes the speed follow_node would command while
    both cars stand (gap already at gap_ref): any command there is estimator noise.
(2) Opponent: stand 3 s, accelerate to 1.5 m/s, cruise, brake at 3 m/s^2, stand, accelerate to
    1.2 m/s, cruise, brake at 6 m/s^2, stand. Optional lateral weave (+-0.15 / +-0.30 m per 6 m).
    The ego runs follow_node's control law on the EKF output; its motor is a 0.15 s lag (assumed, not
    measured). Detections arrive 60 ms late at 15 Hz; their misses, duplicate boxes, lateral offsets
    and range jitter are replayed frame by frame from the recording.
The absolute numbers depend on the assumed motor model; compare parameter sets with each other.
"""
import math, sys

import numpy as np

from replay_ekf_npz import BASE, LASER, MOTION, SUGGESTED, events
from opponent_ekf import EKFParams, OpponentEKF, PHI, PX, VO  # path set by replay_ekf_npz

FOLLOW = dict(ego_axle_to_front=0.45, opp_rear_to_center=0.28, min_gap=0.2, k_gap=1.0, k_close=1.5, k_rel=0.6, max_accel=2.0,
              max_decel=4.0, estop_gap=0.15, gap_ref=0.5, max_speed=1.5, min_track_speed=0.0)
SEGMENTS = [(3.0, 0, 0), (1.0, 0, 1.5), (4.0, 1.5, 1.5), (0.5, 1.5, 0), (3.0, 0, 0), (0.8, 0, 1.2), (4.0, 1.2, 1.2), (0.2, 1.2, 0), (3.0, 0, 0)]
T_END = sum(s[0] for s in SEGMENTS)


def target(px, phi, vo, v_ego, F):
    gap = px - F['opp_rear_to_center'] - F['ego_axle_to_front']
    v_opp = vo * math.cos(phi)
    if v_opp < F['min_track_speed']: v_opp = 0.0
    err = gap - F['gap_ref']
    return gap, float(np.clip(v_opp + (F['k_gap'] if err >= 0 else F['k_close']) * err + F['k_rel'] * (v_opp - v_ego), 0.0, F['max_speed']))


def standing(d, ekf_over, fol_over):
    ekf = OpponentEKF(EKFParams(**{**BASE, **MOTION['straight'], **ekf_over})); F = {**FOLLOW, **fol_over}
    t_still, bias, t_pub, tg, yaw = None, 0.0, None, [], []
    for t, kind, data in events(d):
        if kind == 'v':
            t_still = (t if t_still is None else t_still) if abs(data) < 0.02 else None; ekf.add_speed(t, data)
        elif kind == 'w':
            if t_still is not None and t - t_still > 0.5: bias += 0.01 * (data - bias)
            ekf.add_yaw_rate(t, data - bias)
        else:
            ekf.add_detections(data[0], data[1], LASER)
        if t_pub is None or t - t_pub >= 0.02:
            t_pub = t; e = ekf.estimate(t)
            if e is not None and e.confirmed and t > 3.0:
                tg.append(target(e.x[PX], e.x[PHI], e.x[VO], 0.0, F)[1]); yaw.append(abs(math.degrees(e.x[PHI])))
    tg = np.array(tg)
    return tg.max(), 100 * (tg > 0.1).mean(), max(yaw)


def opp_speed(t):
    for dur, a, b in SEGMENTS:
        if t < dur: return a + (b - a) * t / dur
        t -= dur
    return 0.0


def simulate(frames, x_med, ekf_over, fol_over, offset, weave=0.0):
    ekf = OpponentEKF(EKFParams(**{**BASE, **MOTION['straight'], **ekf_over})); F = {**FOLLOW, **fol_over}
    rng = np.random.default_rng(offset)
    dt = 0.001; xo, yo, xe, ve, v_cmd = 0.5 + 0.28 + 0.45, 0.0, 0.0, 0.0, 0.0
    pending, k, log, res = [], offset, [], {}
    nxt_odom = nxt_obs = nxt_ctl = 0.0
    for i in range(int(T_END / dt)):
        t = i * dt; vo = opp_speed(t)
        if t >= nxt_odom:
            nxt_odom += 0.02; ekf.add_speed(t, ve + rng.normal(0, 0.01)); ekf.add_yaw_rate(t, rng.normal(0, 0.0022))
        if t >= nxt_obs:
            nxt_obs += 1 / 15.0; fr = frames[k % len(frames)]; k += 1
            surf = (xo - xe) - 0.28 - 0.27
            pending.append((t + 0.06, t, [np.array([surf + (x - x_med), yo + y]) + LASER for x, y in fr]))
        while pending and pending[0][0] <= t:
            _, ts, pts = pending.pop(0); r = ekf.add_detections(ts, pts, LASER); res[r] = res.get(r, 0) + 1
        if t >= nxt_ctl:
            nxt_ctl += 0.02; e = ekf.estimate(t); gap_true = (xo - xe) - 0.28 - 0.45
            fresh = e is not None and e.confirmed; tgt, vhat, gap_est = 0.0, math.nan, math.nan
            if fresh: gap_est, tgt = target(e.x[PX], e.x[PHI], e.x[VO], ve, F); vhat = e.x[VO] * math.cos(e.x[PHI])
            if gap_true < F['estop_gap'] or (fresh and gap_est < F['min_gap']): v_cmd = 0.0
            else: v_cmd += float(np.clip(tgt - v_cmd, -F['max_decel'] * 0.02, F['max_accel'] * 0.02))
            log.append((t, vo, vhat, gap_true, ve))
        ve = max(ve + np.clip((v_cmd - ve) / 0.15, -6.0, 4.0) * dt, 0.0)
        xe += ve * dt; xo += vo * dt; yo += vo * dt * weave * (2 * math.pi / 6.0) * math.cos(2 * math.pi * xo / 6.0)
    t, vo, vh, gap, vego = np.array(log).T
    ok = np.isfinite(vh); mov = ok & (vo > 0); stand = (vo == 0) & (t > 1.5) & ((t < 3) | ((t > 10.5) & (t < 11.5)) | (t > 18.0))
    return dict(rms=float(np.sqrt(np.mean((vh[mov] - vo[mov]) ** 2))), gmin=float(gap.min()), creep=float(vego[stand].max()),
                lost=100 * float((~ok[t > 1]).mean()), reinit=res.get('reinit', 0))


SETS = [
    ('launch (motion:=straight)', {}, {}),
    ('suggested', SUGGESTED, {}),
    ('suggested + min_track_speed 0.1', SUGGESTED, {'min_track_speed': 0.1}),
    ('suggested + opp_accel_std 1.0', {**SUGGESTED, 'opp_accel_std': 1.0}, {}),
]


def main():
    d = np.load(sys.argv[1])
    first = np.searchsorted(d['det_frame'], np.arange(len(d['obs_n'])))
    frames = [[(float(d['det_x'][j]), float(d['det_y'][j])) for j in range(first[i], first[i] + int(d['obs_n'][i])) if d['det_conf'][j] >= 0.5] for i in range(len(d['obs_n']))]
    for f in frames: f.sort(key=lambda p: 0)          # order as recorded (most confident first on the car)
    x_med = float(np.median(d['det_x']))
    print(f"{'':34s}| standing recording          | straight sim (3 noise offsets)   | weave +-0.30 m")
    print(f"{'parameters':34s}| cmd max  >0.1 m/s  |yaw|max | v rms  gap min  creep  lost%   | v rms  gap min  lost%  reinit")
    for name, eo, fo in SETS:
        s = standing(d, eo, fo); out = f"{name:34s}|  {s[0]:5.2f}    {s[1]:3.0f}%     {s[2]:5.0f} |"
        for wv in (0.0, 0.30):
            R = [simulate(frames, x_med, eo, fo, o, wv) for o in (0, 97, 211)]
            m = lambda key: float(np.mean([r[key] for r in R]))
            out += (f" {m('rms'):5.3f}  {min(r['gmin'] for r in R):6.3f}  {max(r['creep'] for r in R):5.3f}  {max(r['lost'] for r in R):4.1f}   |" if wv == 0.0 else
                    f" {m('rms'):5.3f}  {min(r['gmin'] for r in R):6.3f}  {max(r['lost'] for r in R):4.1f}   {sum(r['reinit'] for r in R)}")
        print(out, flush=True)


if __name__ == '__main__':
    main()
