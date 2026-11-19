#!/usr/bin/env python3
"""Minimum-lap-time raceline for a map, from any closed line that stays inside the track.

    python3 scripts/optimize_raceline.py                      # race 2 map, from race2_raceline.csv
    python3 scripts/optimize_raceline.py --map <map.yaml> --start <line.csv> --out <new.csv>

The line is moved sideways (offset d(s), a periodic cubic spline through a knot every ~0.6 m)
to minimize the lap time of the speed profile the car drives:

    v = min(sqrt(a_lat / |kappa|), v_max), accelerating / braking with at most a_lon

under penalties that keep it drivable:
    clearance   every point >= `clearance` m from the nearest wall of the map
    steering    curvature within what the servo can do (left / right limits differ on the car)
    smoothness  a small cost on curvature change, so the line does not wiggle

Each round moves the line by at most `step` m and then re-measures from the new line, so the
offsets stay small compared to the corner radii. The start line only has to be inside the track
(the old raceline, a recorded lap, a centerline); it does not have to be good.

Output: x_m, y_m, yaw_rad, v_mps every 0.1 m (v is this script's profile; mppi_node.py builds its
own from the curvature with its a_lat / a_lon / v_max parameters) and a picture next to it.
"""
import argparse
import pathlib
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import distance_transform_edt, map_coordinates

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'mppi' / 'scripts'))
sys.path.insert(0, str(HERE))
import lines  # noqa: E402
from make_track_bounds import load_map  # noqa: E402


class Problem:
    def __init__(self, map_yaml, args):
        free, self.res, self.org = load_map(map_yaml)
        self.dist = (distance_transform_edt(free) * self.res).astype(float)
        self.a = args
        self.limits = dict(a_lat=args.a_lat, a_lon=args.a_lon, v_min=args.v_min, v_max=args.v_max)
        self.k_left = np.tan(args.steer_left) / 0.33
        self.k_right = np.tan(args.steer_right) / 0.33

    def clearance(self, xy):
        """distance to the nearest wall, interpolated between map cells (smooth for the optimizer)"""
        c = (xy[:, 0] - self.org[0]) / self.res - 0.5
        r = (xy[:, 1] - self.org[1]) / self.res - 0.5
        return map_coordinates(self.dist, [r, c], order=1, mode='nearest')

    def lap(self, xy):
        v = lines.speed_profile(xy, **self.limits)
        seg = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)
        return float(np.sum(seg / np.maximum(0.5 * (v + np.roll(v, -1)), 1e-3))), v

    def report(self, xy):
        t, v = self.lap(xy)
        kap = lines.curvature(xy, smooth=0.2)
        seg = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)
        return dict(lap=t, length=float(seg.sum()), v_min=float(v.min()), v_mean=float(seg.sum() / t),
                    clearance=float(self.clearance(xy).min()), k_left=float(kap.max()), k_right=float(-kap.min()))

    def cost(self, xy):
        a = self.a
        t, _ = self.lap(xy)
        kap = lines.curvature(xy, smooth=0.2)
        seg = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)
        wall = np.maximum(a.clearance - self.clearance(xy), 0.0)
        steer = np.maximum(kap - 0.97 * self.k_left, 0.0) + np.maximum(-kap - 0.97 * self.k_right, 0.0)
        dk = np.roll(kap, -1) - kap
        return (t + a.w_wall * float(np.sum(wall ** 2 * seg)) + a.w_steer * float(np.sum(steer ** 2 * seg))
                + a.w_smooth * float(np.sum(dk ** 2)))


def spline_offsets(n, knots):
    """periodic cubic spline through the knot values, at n points around the lap"""
    k = len(knots)
    cs = CubicSpline(np.arange(k + 1) / k, np.r_[knots, knots[:1]], bc_type='periodic')
    return cs(np.arange(n) / n)


def optimize(prob, xy, args):
    # Coarse to fine: a knot every few meters first (moves a whole corner sideways, which is where
    # the time is), then closer knots for the details. With close knots from the start nothing
    # moves: shifting one knot alone only adds a bump.
    spacings = [s for s in (4.0, 3.0, 2.0, 1.5, 1.0) if s > args.knot_spacing] + [args.knot_spacing]
    schedule = [s for s in spacings for _ in range(args.rounds)]
    quiet = 0
    for rnd, spacing in enumerate(schedule):
        xy = lines.resample_closed(xy)
        n = len(xy)
        length = n * 0.1
        k = max(int(round(length / spacing)), 6)
        t0 = time.time()

        def f(knots, xy=xy, n=n):
            return prob.cost(lines.offset_line(xy, spline_offsets(n, knots)))

        # Pattern search: try each knot a step to either side, keep what lowers the cost, halve the
        # step when a whole sweep finds nothing. (A gradient method stalls at once here: the lap
        # time of a speed profile with limits is not smooth in the line.)
        x, fx, evals = np.zeros(k), None, 0
        fx = f(x)
        step = args.step / 2.0
        while step >= args.min_step:
            improved = False
            for j in np.random.default_rng(rnd).permutation(k):
                for sgn in (1.0, -1.0):
                    y = x.copy()
                    y[j] = np.clip(y[j] + sgn * step, -args.step, args.step)
                    fy = f(y)
                    evals += 1
                    if fy < fx - 1e-5:
                        x, fx, improved = y, fy, True
                        break
            if not improved:
                step *= 0.5
        res = type('R', (), dict(x=x, nit=evals))
        xy = lines.offset_line(xy, spline_offsets(n, res.x))
        r = prob.report(lines.resample_closed(xy))
        print(f'round {rnd + 1} (knot every {spacing} m): lap {r["lap"]:.3f} s, length {r["length"]:.2f} m, clearance {r["clearance"]:.2f} m, '
              f'max |offset| {np.abs(res.x).max():.2f} m, {res.nit} evaluations, {time.time() - t0:.0f} s', flush=True)
        # this spacing has nothing left: skip its remaining rounds
        quiet = quiet + 1 if np.abs(res.x).max() < 0.02 else 0
        if quiet and rnd + 1 < len(schedule) and schedule[rnd + 1] == spacing:
            schedule[rnd + 1:] = [s for s in schedule[rnd + 1:] if s != spacing]
            if rnd + 1 >= len(schedule):
                break
    return lines.resample_closed(xy)


def picture(prob, old, new, v_new, path):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    rr, cc = np.nonzero(prob.dist > 0)
    x0, x1 = prob.org[0] + cc.min() * prob.res - 0.3, prob.org[0] + cc.max() * prob.res + 0.3
    y0, y1 = prob.org[1] + rr.min() * prob.res - 0.3, prob.org[1] + rr.max() * prob.res + 0.3
    k = 90
    img = Image.new('RGB', (int((x1 - x0) * k), int((y1 - y0) * k)), (150, 150, 150))
    dr = ImageDraw.Draw(img)
    P = lambda x, y: ((x - x0) * k, (y1 - y) * k)
    for r, c in zip(rr, cc):
        x, y = prob.org[0] + c * prob.res, prob.org[1] + r * prob.res
        dr.rectangle([P(x, y + prob.res), P(x + prob.res, y)], fill=(255, 255, 255))
    pts = [P(*p) for p in old]
    dr.line(pts + pts[:1], fill=(0, 0, 0), width=2)
    lo, hi = v_new.min(), v_new.max()
    for i in range(len(new)):  # new line colored by speed: red = slow, green = fast
        f = (v_new[i] - lo) / max(hi - lo, 1e-6)
        dr.line([P(*new[i]), P(*new[(i + 1) % len(new)])], fill=(int(255 * (1 - f)), int(200 * f), 0), width=5)
    for i in range(0, len(new), 20):
        p = P(*new[i])
        dr.text((p[0] + 6, p[1] - 6), f'{i * 0.1:.0f}', fill=(0, 0, 255))
    img.save(path)
    print(f'wrote {path} (black = start line, colored = optimized: red slow, green fast; numbers = station [m])')


def main():
    root = HERE.parent / 'mppi'
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--map', default=str(root / 'maps' / 'race2_map_edited.yaml'))
    ap.add_argument('--start', default=str(root / 'waypoints' / 'race2_raceline.csv'))
    ap.add_argument('--out', default=str(root / 'waypoints' / 'race2_raceline_opt.csv'))
    # the speed model: the limits the team's final race 2 raceline was built with (scripts/raceline.py)
    ap.add_argument('--a-lat', type=float, default=4.3)
    ap.add_argument('--a-lon', type=float, default=2.5)
    ap.add_argument('--v-min', type=float, default=1.0)
    ap.add_argument('--v-max', type=float, default=4.0)
    # what keeps it drivable
    ap.add_argument('--clearance', type=float, default=0.45, help='[m] line to the nearest wall')
    ap.add_argument('--steer-left', type=float, default=0.32, help='[rad] servo limit, left')
    ap.add_argument('--steer-right', type=float, default=0.25, help='[rad] servo limit, right')
    ap.add_argument('--w-wall', type=float, default=400.0)
    ap.add_argument('--w-steer', type=float, default=40.0)
    ap.add_argument('--w-smooth', type=float, default=0.1)
    # the search
    ap.add_argument('--knot-spacing', type=float, default=0.6)
    ap.add_argument('--step', type=float, default=0.35, help='[m] largest sideways move per round')
    ap.add_argument('--rounds', type=int, default=6, help='rounds per knot spacing, at most')
    ap.add_argument('--min-step', type=float, default=0.01, help='[m] finest sideways move tried')
    args = ap.parse_args()

    prob = Problem(args.map, args)
    start = lines.resample_closed(np.loadtxt(args.start, delimiter=',', comments='#', ndmin=2)[:, :2])
    print(f'speed model: a_lat {args.a_lat}, a_lon {args.a_lon} m/s^2, v {args.v_min}-{args.v_max} m/s; '
          f'clearance >= {args.clearance} m; curvature <= {prob.k_left:.2f} left / {prob.k_right:.2f} right 1/m')
    new = optimize(prob, start, args)

    print(f'\n{"":10s} {"lap [s]":>8s} {"length":>7s} {"v mean":>7s} {"v min":>6s} {"clearance":>10s} {"kappa L":>8s} {"kappa R":>8s}')
    for name, xy in (('start', start), ('optimized', new)):
        r = prob.report(xy)
        print(f'{name:10s} {r["lap"]:8.2f} {r["length"]:7.2f} {r["v_mean"]:7.2f} {r["v_min"]:6.2f} {r["clearance"]:10.2f} '
              f'{r["k_left"]:8.2f} {r["k_right"]:8.2f}')
    _, v = prob.lap(new)
    yaw = lines.heading(new)
    with open(args.out, 'w') as f:
        f.write('# x_m,y_m,yaw_rad,v_mps\n')
        f.write(f'# minimum lap time on {pathlib.Path(args.map).name} from {pathlib.Path(args.start).name} '
                f'(scripts/optimize_raceline.py): a_lat {args.a_lat}, a_lon {args.a_lon}, v_max {args.v_max}, '
                f'clearance {args.clearance}\n')
        f.writelines(f'{x:.4f},{y:.4f},{a:.4f},{s:.4f}\n' for (x, y), a, s in zip(new, yaw, v))
    print(f'wrote {args.out} ({len(new)} points)')
    picture(prob, start, new, v, str(pathlib.Path(args.out).with_suffix('.png')))


if __name__ == '__main__':
    main()
