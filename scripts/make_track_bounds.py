#!/usr/bin/env python3
"""Measure the free track width left and right of a reference line from the map.

    python3 scripts/make_track_bounds.py mppi/maps/race2_map_edited.yaml \
        mppi/waypoints/race2_raceline.csv

Writes <map>_centerline.csv next to the map yaml:

    # x_m, y_m, w_tr_right_m, w_tr_left_m

one row every 0.1 m along the line. Two users:
  - the simulator (f1tenth_gym_ros) picks the file up by its name and uses it as the Frenet
    frame for lap counting / lap times,
  - the head-to-head planner reads the widths to know how much room there is on each side
    of the raceline (overtaking side, switchback line).
The line does not have to be the geometric centerline: the widths are measured from it.
"""
import argparse
import pathlib

import numpy as np
import yaml
from PIL import Image


def load_map(yaml_path):
    """free (H, W) bool with row 0 at the BOTTOM (y up), resolution [m], origin (x, y)"""
    yaml_path = pathlib.Path(yaml_path)
    spec = yaml.safe_load(yaml_path.read_text())
    img = np.asarray(Image.open(yaml_path.parent / spec['image']).convert('L'), dtype=np.float32)
    occ = (255.0 - img) / 255.0 if not spec.get('negate', 0) else img / 255.0
    free = occ < float(spec.get('free_thresh', 0.196))  # unknown and occupied both count as wall
    return free[::-1].copy(), float(spec['resolution']), tuple(spec['origin'][:2])


def resample_closed(xy, ds=0.1):
    xy_c = np.vstack([xy, xy[:1]])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy_c, axis=0).T))]
    n = int(s[-1] / ds)
    s_new = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.c_[np.interp(s_new, s, xy_c[:, 0]), np.interp(s_new, s, xy_c[:, 1])]


def measure_widths(xy, free, res, origin, w_max=5.0, step=0.01):
    """distance [m] from every point to the first non-free cell along its right / left normal"""
    yaw = np.arctan2(np.roll(xy[:, 1], -1) - np.roll(xy[:, 1], 1),
                     np.roll(xy[:, 0], -1) - np.roll(xy[:, 0], 1))
    nx, ny = -np.sin(yaw), np.cos(yaw)  # left normal
    d = np.arange(0.0, w_max, step)

    def march(sign):
        px = xy[:, 0:1] + sign * d[None, :] * nx[:, None]
        py = xy[:, 1:2] + sign * d[None, :] * ny[:, None]
        c = np.floor((px - origin[0]) / res).astype(int)
        r = np.floor((py - origin[1]) / res).astype(int)
        inside = (r >= 0) & (r < free.shape[0]) & (c >= 0) & (c < free.shape[1])
        ok = np.zeros(px.shape, dtype=bool)
        ok[inside] = free[r[inside], c[inside]]
        blocked = ~ok
        first = np.where(blocked.any(axis=1), blocked.argmax(axis=1), len(d) - 1)
        return d[first]

    return march(-1.0), march(+1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('map_yaml')
    ap.add_argument('line_csv', help='x, y in the first two columns (map frame), closed loop')
    ap.add_argument('--out', default=None, help='default: <map>_centerline.csv next to the yaml')
    args = ap.parse_args()

    free, res, origin = load_map(args.map_yaml)
    xy = resample_closed(np.loadtxt(args.line_csv, delimiter=',', comments='#', ndmin=2)[:, :2])
    w_r, w_l = measure_widths(xy, free, res, origin)
    yaml_path = pathlib.Path(args.map_yaml)
    out = pathlib.Path(args.out) if args.out else yaml_path.with_name(yaml_path.stem + '_centerline.csv')
    length = len(xy) * 0.1
    with open(out, 'w') as f:
        f.write('# x_m, y_m, w_tr_right_m, w_tr_left_m\n')
        f.write(f'# {len(xy)} points, {length:.1f} m, from {pathlib.Path(args.line_csv).name}; '
                f'widths measured from {yaml_path.name} by scripts/make_track_bounds.py\n')
        for (x, y), r, l in zip(xy, w_r, w_l):
            f.write(f'{x:.4f}, {y:.4f}, {r:.3f}, {l:.3f}\n')
    print(f'{out}: {len(xy)} points, {length:.1f} m, right {w_r.min():.2f}-{w_r.max():.2f} m, '
          f'left {w_l.min():.2f}-{w_l.max():.2f} m, total width {(w_r + w_l).min():.2f}-'
          f'{(w_r + w_l).max():.2f} m')


if __name__ == '__main__':
    main()
