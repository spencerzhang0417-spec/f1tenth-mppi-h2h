#!/usr/bin/env python3
"""Detection rate, confidence and bearing of the lab 10 observations in the recorded runs. numpy only.

    python3 detection_stats.py ../npz/*.npz

Per file: how many frames carry a car, and the same split by the distance of the nearest LiDAR return
straight ahead of the ego (a corridor of +-0.2 m; returns closer than 0.1 m are ignored: the scan has
stray returns at 0.03-0.07 m). The bearing range tells how much of the camera's +-35 deg was used.
"""
import sys

import numpy as np

EDGES = [0.0, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 100.0]


def main():
    for path in sys.argv[1:]:
        d = np.load(path)
        n = d['obs_n']; first = np.searchsorted(d['det_frame'], np.arange(len(n)))
        best = np.array([first[i] + int(np.argmax(d['det_conf'][first[i]:first[i] + n[i]])) if n[i] else -1 for i in range(len(n))])
        has = best >= 0
        r = d['scan_ranges']; ang = float(d['scan_angle_min']) + float(d['scan_angle_increment']) * np.arange(r.shape[1])
        x, y = r * np.cos(ang), r * np.sin(ang)
        front = np.where((r > 0.1) & (x > 0) & (np.abs(y) < 0.2), x, np.inf).min(axis=1)
        fr = np.interp(d['obs_t'], d['scan_t'], np.where(np.isfinite(front), front, 99.0))
        bx, by, bc = d['det_x'][best[has]], d['det_y'][best[has]], d['det_conf'][best[has]]
        print(f"\n{str(d['bag'])}: {len(n)} frames at {(len(n) - 1) / (d['obs_t'][-1] - d['obs_t'][0]):.1f} Hz, {100 * has.mean():.0f}% with a car, "
              f"{100 * np.mean(n > 1):.0f}% with more than one detection | ego |v| max {np.abs(d['odom_v']).max():.2f} m/s")
        if has.any():
            print(f"   best detection: range {np.hypot(bx, by).min():.2f}..{np.hypot(bx, by).max():.2f} m, bearing {np.degrees(np.arctan2(by, bx)).min():+.1f}..{np.degrees(np.arctan2(by, bx)).max():+.1f} deg, "
                  f"confidence {bc.mean():.2f} ({bc.min():.2f}..{bc.max():.2f}) | observation age (receive - scan stamp) {1e3 * np.mean(d['obs_t'] - d['obs_stamp']):.0f} ms")
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            s = (fr >= lo) & (fr < hi)
            if s.sum() >= 10:
                print(f"   nearest return ahead {lo:3.1f}-{hi:5.1f} m: {s.sum():5d} frames, detected {100 * has[s].mean():3.0f}%")


if __name__ == '__main__':
    main()
