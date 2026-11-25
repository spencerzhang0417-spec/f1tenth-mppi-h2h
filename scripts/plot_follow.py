#!/usr/bin/env python3
"""Plot and score a follow experiment log (follow_node.py CSV).

    python3 scripts/plot_follow.py ~/follow_logs/follow_20261124_153000.csv     # -> <csv>.png

Ground truth for the opponent's speed, best first:
    v_opp_truth   the opponent's own VESC speed (follow_node truth_topic), if it was logged
    v_ego + d(gap_lidar)/dt   the raw LiDAR gap straight ahead, differentiated (smoothed 0.3 s):
                  independent of the camera and the filter, valid while the opponent is the
                  nearest thing in the corridor ahead
Scored only while following (state 'follow') and moving (v_ego > 0.3 m/s).
"""
import csv
import sys

import numpy as np


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != 'state'}
    out['state'] = np.array([r['state'] for r in rows])
    out['t'] -= out['t'][0]
    return out


def smooth(x, n):
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(np.nan_to_num(x, nan=0.0, posinf=0.0), k, mode='same')


def main(path):
    d = load(path)
    t = d['t']
    dt = np.median(np.diff(t))
    n = max(1, int(round(0.3 / dt)))
    gl = np.where(np.isfinite(d['gap_lidar']) & (d['gap_lidar'] < 5.0), d['gap_lidar'], np.nan)
    v_lidar = d['v_ego'] + np.gradient(smooth(gl, n), t)
    v_lidar[~np.isfinite(gl)] = np.nan
    has_truth = np.isfinite(d['v_opp_truth']).any()
    v_true = d['v_opp_truth'] if has_truth else v_lidar
    use = (d['state'] == 'follow') & (d['v_ego'] > 0.3)

    print(f'{path}: {t[-1]:.1f} s, following {use.mean() * 100:.0f} % of the time')
    for s in np.unique(d['state']):
        print(f'  {s:12s} {np.mean(d["state"] == s) * 100:5.1f} %')
    if use.any():
        g = d['gap'][use]
        e_v = (d['v_opp'] - v_true)[use]
        e_v = e_v[np.isfinite(e_v)]
        print(f'  gap (filter)  mean {np.mean(g):.3f} m, std {np.std(g):.3f}, min {np.min(g):.3f}, max {np.max(g):.3f}')
        gl_u = gl[use][np.isfinite(gl[use])]
        if gl_u.size:
            print(f'  gap (LiDAR)   mean {np.mean(gl_u):.3f} m, std {np.std(gl_u):.3f}, min {np.min(gl_u):.3f}')
            print(f'  filter - LiDAR gap: mean {np.nanmean((d["gap"] - gl)[use]):+.3f} m '
                  '(a constant offset = ego_axle_to_front / opp_rear_to_center are off)')
        if e_v.size:
            print(f'  opponent speed error vs {"truth_topic" if has_truth else "LiDAR gap rate"}: '
                  f'mean {np.mean(e_v):+.3f} m/s, rms {np.sqrt(np.mean(e_v ** 2)):.3f} m/s')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
    ax[0].plot(t, d['gap'], label='gap (filter)')
    ax[0].plot(t, gl, '.', ms=2, alpha=0.5, label='gap (raw LiDAR ahead)')
    ax[0].set_ylabel('gap [m]')
    ax[0].set_ylim(0, 2.5)
    ax[1].plot(t, d['v_ego'], label='ego (VESC)')
    ax[1].plot(t, d['v_opp'], label='opponent (EKF)')
    ax[1].fill_between(t, d['v_opp'] - d['v_opp_sd'], d['v_opp'] + d['v_opp_sd'], alpha=0.2, color='C1')
    ax[1].plot(t, v_true, alpha=0.6, label='opponent (truth_topic)' if has_truth else 'opponent (ego + LiDAR gap rate)')
    ax[1].plot(t, d['v_cmd'], '--', lw=1, label='command')
    ax[1].set_ylabel('speed [m/s]')
    ax[2].plot(t, d['opp_y'], label='opponent y (EKF)')
    ax[2].plot(t, d['steer'], label='steering [rad]')
    ax[2].set_xlabel('t [s]')
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(loc='upper right', fontsize=8)
    # shade the stops
    for a in ax:
        stopped = d['state'] != 'follow'
        a.fill_between(t, 0, 1, where=stopped, transform=a.get_xaxis_transform(), color='red', alpha=0.08)
    fig.suptitle(f'{path}  (red: stopped / no track)', fontsize=9)
    fig.tight_layout()
    out = path.rsplit('.', 1)[0] + '.png'
    fig.savefig(out, dpi=120)
    print(f'  plot: {out}')


if __name__ == '__main__':
    main(sys.argv[1])
