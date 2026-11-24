#!/usr/bin/env python3
"""Late-apex and early-apex lines for every tight corner of a raceline, and how they compare.

    python3 scripts/make_apex_lines.py                 # race2_raceline_opt.csv on the race 2 map

  late apex    the attacker's ("switchback") line. For a car that covers the inside the raceline
               is blocked: its apex is where the other car is. This is the fastest line that
               stays OUTSIDE of the raceline on the way in (`apex_room` m, room for the car
               covering the inside), then cuts back and hugs the inside at a LATER apex than the
               raceline does: wide in, deep, late turn, straight out.
  early apex   the defender's line: the fastest line that COVERS THE INSIDE on the way in
               (`entry_inside` m inside of the raceline until shortly before the apex). It
               turns in early, cannot hold the tight line and runs wide on the way out.

Both are found per corner with the same lap-time model, wall clearance and steering limits as
the raceline itself (scripts/optimize_raceline.py), so the three lines compare like for like.
Corners are found from the raceline's curvature (|kappa| > `corner_kappa`). Every line is a
sideways offset from the raceline per waypoint, zero outside the corners, so it is index-aligned
with the raceline: switching lines is switching arrays (mppi/scripts/lines.py).

Output next to the raceline:
  <raceline>_apex.json        offsets 'attack' (late) and 'defend' (early) + the corners
  <raceline>_lateapex.csv     each as a whole lap of its own: x_m, y_m, yaw_rad, v_mps
  <raceline>_earlyapex.csv
  <raceline>_apex.png         picture: the three lines, and both cars every 0.5 s with the
                              attacker starting `gap` m behind the defender

The times are for perfect tracking of the speed profile; the simulator is the test.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.interpolate import CubicSpline

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'mppi' / 'scripts'))
sys.path.insert(0, str(HERE))
import lines  # noqa: E402
from optimize_raceline import Problem  # noqa: E402


def find_corners(kappa, ds, k_min, merge=1.0):
    """[(first index, last index, apex index, inside sign)] of the stretches with |kappa| > k_min"""
    n = len(kappa)
    hot = np.abs(kappa) > k_min
    if not hot.any() or hot.all():
        return []
    start = int(np.argmin(hot))  # a point outside every corner: no corner wraps around it
    idx = (start + np.arange(n)) % n
    runs, cur = [], None
    for i in idx:
        if hot[i]:
            cur = [i, i] if cur is None else [cur[0], i]
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    merged = [runs[0]]
    for a, b in runs[1:]:
        if ((a - merged[-1][1]) % n) * ds < merge and np.sign(kappa[a]) == np.sign(kappa[merged[-1][1]]):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        seg = (a + np.arange((b - a) % n + 1)) % n
        apex = int(seg[np.argmax(np.abs(kappa[seg]))])
        out.append((a, b, apex, int(np.sign(kappa[apex]))))
    return out


class Corner:
    """one corner and one kind of line through it: the zone the line may change in and where it
    has to be -- 'late': outside of the raceline until the apex; 'early': inside of it on the way in"""
    def __init__(self, prob, xy, apex, inside, args, kind, d_def=None):
        self.n, self.ds = len(xy), 0.1
        self.apex, self.inside, self.kind = apex, inside, kind
        self.before = args.before if kind == 'late' else args.early_before
        self.i0 = int(apex - self.before / self.ds) % self.n       # lines split here
        after = args.after if kind == 'late' else (args.early_after or args.after)
        m = int((self.before + after) / self.ds)
        self.idx = (self.i0 + np.arange(m + 1)) % self.n            # the zone, in driving order
        self.rel = np.arange(m + 1) * self.ds                       # station from the start of the zone
        to_apex = self.before - self.rel                            # distance still to go to the apex
        # where the line has to be, as bounds on u = offset towards the INSIDE of the turn [m]:
        # u <= hi (stay out), u >= lo (stay in); +-inf where it is free
        yaw = lines.heading(xy)
        dd = np.arange(0.0, 3.0, 0.02)

        def room(side):
            """free distance from the raceline towards side (+1 inside, -1 outside), per zone point"""
            out = np.empty(len(self.idx))
            for j, i in enumerate(self.idx):
                p = np.c_[xy[i, 0] - side * inside * dd * np.sin(yaw[i]), xy[i, 1] + side * inside * dd * np.cos(yaw[i])]
                c = prob.clearance(p)
                out[j] = dd[np.argmax(c < prob.a.clearance)] if (c < prob.a.clearance).any() else dd[-1]
            return np.maximum(out - 0.03, 0.0)   # never ask for more than the track has

        self.lo, self.hi = np.full(len(self.idx), -np.inf), np.full(len(self.idx), np.inf)
        if kind == 'late':
            # Where the line has to be is given by where the DEFENDER is (d_def, its line): only
            # as far out as it takes to leave the inside to that car. (It used to be "0.6 m outside
            # of the raceline": in a wide corner, where the raceline comes in from the outside
            # anyway and the defender is 1.5 m further in, that was a detour of 0.3 s for nothing.)
            u_def = inside * np.asarray(d_def)[self.idx]
            # on the way in: apex_room OUTSIDE of the defender's line, from ramp_from -> ramp_to m
            # before the apex, released `release` m before it ...
            ramp = np.clip((args.ramp_from - to_apex) / (args.ramp_from - args.ramp_to), 0.0, 1.0)
            m_in = (to_apex <= args.ramp_from) & (to_apex >= args.release)
            self.hi[m_in] = np.maximum(u_def - args.apex_room * ramp, -room(-1))[m_in]
            # ... and on the way out INSIDE of it, from exit_from m past the apex until the lines
            # are together again (the room tapers off over the last 2 m). This is the cut-back:
            # the defender runs wide out of its early apex, the attacker turns late and comes out
            # between it and the inside wall. Without it the fastest "wide in" line runs a wide
            # arc around the whole corner, into the car it is passing.
            m_out = to_apex <= -args.exit_from
            taper = np.clip((self.rel[-1] - self.rel) / args.exit_taper, 0.0, 1.0)
            taper = taper * taper * (3.0 - 2.0 * taper)
            self.lo[m_out] = np.minimum((u_def + args.apex_room) * taper, room(+1))[m_out]
            # (and at least exit_inside inside of the raceline shortly after the apex)
            m_x = (to_apex <= -args.exit_from) & (to_apex >= -args.exit_to)
            self.lo[m_x] = np.maximum(self.lo, np.minimum(args.exit_inside, room(+1)))[m_x]
        else:
            # covers the inside on the way in: in by entry_inside from early_from -> early_full m
            # before the apex, free again early_release m before it
            ramp = np.clip((args.early_from - to_apex) / (args.early_from - args.early_full), 0.0, 1.0)
            m_in = (to_apex <= args.early_from) & (to_apex >= args.early_release)
            # (the ramp on what the track allows, not on entry_inside: with "as far in as the wall
            # lets it" asked for, the whole move happened in the first fifth of the ramp, a lane
            # change the defender braked from 6 to 3 m/s for)
            ramp = ramp * ramp * (3.0 - 2.0 * ramp)
            self.lo[m_in] = (np.minimum(args.entry_inside, room(+1)) * ramp)[m_in]
            # ... and on the way out it comes back to the raceline from the outside, where its
            # early apex has taken it, without crossing to the inside of it: left free there, the
            # fastest line swung from the outside wall over to 0.5 m inside of the raceline, across
            # the lane of the car that had just come up on its inside.
            self.hi[to_apex <= -args.early_out] = 0.0

    def offsets(self, knots_rel, values):
        """offsets for the whole lap from knot values inside the zone (zero at both ends)"""
        cs = CubicSpline(knots_rel, values, bc_type='clamped')
        d = np.zeros(self.n)
        d[self.idx] = cs(self.rel)
        return d

    def violation(self, d):
        """how far the line is from where it has to be [m], per zone point"""
        u = self.inside * d[self.idx]
        return np.maximum(u - self.hi, 0.0) + np.maximum(self.lo - u, 0.0)


def optimize_corner(prob, xy, corner, args):
    d = np.zeros(len(xy))

    def cost(dd):
        v = corner.violation(dd)
        return prob.cost(lines.offset_line(xy, dd)) + args.w_room * float(np.sum(v ** 2) * corner.ds)

    best = cost(d)
    for spacing in (2.0, 1.0, 0.6):                    # coarse to fine, as in optimize_raceline.py
        m = max(int(round(corner.rel[-1] / spacing)), 3)
        knots = np.linspace(0.0, corner.rel[-1], m + 1)
        for _ in range(args.rounds):
            x = np.zeros(m + 1)                        # an increment on top of the line so far
            step, moved = args.step, False
            while step >= args.min_step:
                improved = False
                for j in range(1, m):                  # the end knots stay 0: the line rejoins
                    for sgn in (1.0, -1.0):
                        y = x.copy()
                        y[j] += sgn * step
                        c = cost(d + corner.offsets(knots, y))
                        if c < best - 1e-5:
                            x, best, improved, moved = y, c, True, True
                            break
                if not improved:
                    step *= 0.5
            d = d + corner.offsets(knots, x)
            if not moved:
                break
    return d


def main():
    root = HERE.parent / 'mppi'
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--map', default=str(root / 'maps' / 'race2_map_edited.yaml'))
    ap.add_argument('--raceline', default=str(root / 'waypoints' / 'race2_raceline_opt.csv'))
    # the same model as optimize_raceline.py
    ap.add_argument('--a-lat', type=float, default=4.3)
    ap.add_argument('--a-lon', type=float, default=2.5)
    ap.add_argument('--v-min', type=float, default=1.0)
    ap.add_argument('--v-max', type=float, default=4.0)
    ap.add_argument('--clearance', type=float, default=0.45)
    ap.add_argument('--steer-left', type=float, default=0.32)
    ap.add_argument('--steer-right', type=float, default=0.25)
    ap.add_argument('--w-wall', type=float, default=400.0)
    ap.add_argument('--w-steer', type=float, default=40.0)
    ap.add_argument('--w-smooth', type=float, default=0.1)
    # the late apex
    ap.add_argument('--corner-kappa', type=float, default=0.6, help='[1/m] a corner is where |kappa| exceeds this')
    ap.add_argument('--corners', default='', help='<track>_corners.json of scripts/make_test_track.py: these corners, not the ones found by curvature')
    ap.add_argument('--apex-room', type=float, default=0.6, help="[m] next to the defender's line: outside of it on the way in, inside on the way out")
    ap.add_argument('--before', type=float, default=5.0, help='[m] the lines split this far before the apex')
    ap.add_argument('--after', type=float, default=5.0, help='[m] and are together again this far after it')
    ap.add_argument('--ramp-from', type=float, default=4.0, help='[m] before the apex: start moving out')
    ap.add_argument('--ramp-to', type=float, default=2.5, help='[m] before the apex: fully outside')
    ap.add_argument('--release', type=float, default=1.2, help='[m] before the apex: free to cut back to the inside')
    ap.add_argument('--exit-taper', type=float, default=2.0,
                    help='[m] the room to the defender on the way out tapers off over this last part of the late line')
    ap.add_argument('--exit-inside', type=float, default=0.35, help='[m] inside of the raceline on the way out')
    ap.add_argument('--exit-from', type=float, default=1.0, help='[m] past the apex: inside from here')
    ap.add_argument('--exit-to', type=float, default=2.5, help='[m] past the apex: ... to here')
    # the early apex
    ap.add_argument('--entry-inside', type=float, default=0.5, help='[m] inside of the raceline on the way in')
    ap.add_argument('--early-before', type=float, default=5.0, help='[m] the early line leaves the raceline this far before the apex')
    ap.add_argument('--early-from', type=float, default=4.0, help='[m] before the apex: start moving in')
    ap.add_argument('--early-full', type=float, default=2.5, help='[m] before the apex: fully inside')
    ap.add_argument('--early-release', type=float, default=1.0, help='[m] before the apex: free again')
    ap.add_argument('--early-out', type=float, default=1.0,
                    help='[m] past the apex: from here on the early line is not inside of the raceline any more')
    ap.add_argument('--early-after', type=float, default=0.0,
                    help='[m] the early line is back on the raceline this far after the apex (0 = as --after)')
    ap.add_argument('--gap', type=float, default=1.5, help='[m] attacker behind the defender, for the comparison')
    ap.add_argument('--w-room', type=float, default=400.0)
    ap.add_argument('--step', type=float, default=0.3)
    ap.add_argument('--min-step', type=float, default=0.01)
    ap.add_argument('--rounds', type=int, default=4)
    args = ap.parse_args()

    prob = Problem(args.map, args)
    xy = lines.resample_closed(np.loadtxt(args.raceline, delimiter=',', comments='#', ndmin=2)[:, :2])
    n, ds = len(xy), 0.1
    kap = lines.curvature(xy, smooth=0.3)
    t_base, v_base = prob.lap(xy)
    tb = lines.lap_times(np.c_[xy, np.zeros(n), v_base])
    names = None
    if args.corners:
        # the corners of a test track (scripts/make_test_track.py), also the ones the raceline
        # takes flat out: there it has no curvature to find them by
        spec = json.loads(pathlib.Path(args.corners).read_text())['corners']
        yaw = lines.heading(xy)
        found, names = [], [c['name'] for c in spec]
        for c in spec:
            ic = int(np.argmin(np.hypot(*(xy - np.asarray(c['xy_mid'])).T)))
            half = int((0.5 * (c['s_to'] - c['s_from']) + 1.0) / ds)
            win = (ic + np.arange(-half, half + 1)) % n
            inside = int(np.sign(np.sum(kap[win])))
            # the apex: where the raceline is closest to the inside wall and turns most
            dd = np.arange(0.0, 4.0, 0.02)
            room = np.array([dd[np.argmax(prob.clearance(np.c_[xy[i, 0] - inside * dd * np.sin(yaw[i]),
                                                               xy[i, 1] + inside * dd * np.cos(yaw[i])]) < 0.03)] for i in win])
            apex = int(win[np.argmin(room - 0.3 * np.abs(kap[win]))])
            found.append((int(win[0]), int(win[-1]), apex, inside))
        print(f'{pathlib.Path(args.raceline).name}: {n} points, lap {t_base:.2f} s, {len(found)} corners from {pathlib.Path(args.corners).name}')
    else:
        found = find_corners(kap, ds, args.corner_kappa)
        print(f'{pathlib.Path(args.raceline).name}: {n} points, lap {t_base:.2f} s, {len(found)} corners with |kappa| > {args.corner_kappa}')

    L = n * ds
    wrap = lambda x: round(float(x % L), 2)
    d_late, d_early, corners, zones = np.zeros(n), np.zeros(n), [], []
    for c, (a, b, apex, inside) in enumerate(found):
        name = names[c] if names else f'corner{c + 1}'
        print(f'\n{name}: apex of the raceline at s = {apex * ds:.1f} m, {"left" if inside > 0 else "right"} turn')
        res = {}
        for kind in ('early', 'late'):          # the defender first: the attacker's line is relative to it
            co = Corner(prob, xy, apex, inside, args, kind, d_def=res['early'][1] if kind == 'late' else None)
            res[kind] = (co, optimize_corner(prob, xy, co, args))
        d_late += res['late'][1]
        d_early += res['early'][1]
        # compare the three lines over the same stretch: from where the first one leaves the
        # raceline to where the last one is back on it
        i0 = res['early'][0].idx[0]
        i1 = max(res['late'][0].idx[-1], res['early'][0].idx[-1], key=lambda i: (i - apex) % n)
        seg = (i0 + np.arange((i1 - i0) % n + 1)) % n
        i_exit = int(apex + 2.0 / ds) % n
        print(f'  {"line":12s} {"time":>6s} {"vs race":>8s} {"slowest":>8s} {"2 m after apex":>15s} {"at apex":>9s} {"clearance":>10s} {"kappa":>6s}'
              f'   (stretch s = {i0 * ds:.1f} -> {i1 * ds:.1f} m)')
        times = {}
        for label, d in (('raceline', np.zeros(n)), ('early apex', res['early'][1]), ('late apex', res['late'][1])):
            w = np.c_[lines.offset_line(xy, d), np.zeros(n), np.zeros(n)]
            w[:, 3] = prob.lap(w[:, :2])[1]
            segl = np.hypot(*(w[np.roll(seg, -1)[:-1], :2] - w[seg[:-1], :2]).T)
            tt = np.r_[0.0, np.cumsum(segl / np.maximum(0.5 * (w[seg[:-1], 3] + w[seg[1:], 3]), 1e-3))]  # time along the stretch
            times[label] = (tt, w)
            side = inside * d[apex]
            print(f'  {label:12s} {tt[-1]:6.2f} {tt[-1] - times["raceline"][0][-1]:+8.2f} {w[seg, 3].min():8.2f} {w[i_exit, 3]:15.2f} '
                  f'{("%.2f in" % side) if side > 0.005 else ("%.2f out" % -side) if side < -0.005 else "-":>9s} '
                  f'{prob.clearance(w[seg, :2]).min():10.2f} {np.abs(lines.curvature(w[:, :2], smooth=0.2)[seg]).max():6.2f}')
        near = (apex + np.arange(int(-2.5 / ds), int(3.5 / ds))) % n          # around the corner
        yaw_r = lines.heading(xy)
        for label in ('raceline', 'early apex', 'late apex'):
            w = times[label][1]
            # distance to the wall on the INSIDE of the turn: walk from the line towards the inside
            best = (9.9, 0)
            for i in near:
                dd = np.arange(0.0, 2.5, 0.02)
                p = np.c_[w[i, 0] - inside * dd * np.sin(yaw_r[i]), w[i, 1] + inside * dd * np.cos(yaw_r[i])]
                cl = prob.clearance(p)
                gap_in = dd[np.argmax(cl < 0.03)] if (cl < 0.03).any() else 9.9
                if gap_in < best[0]:
                    best = (gap_in, i)
            shift = ((best[1] - apex + n // 2) % n - n // 2) * ds
            print(f'  {label:12s} hugs the inside at s = {best[1] * ds:.1f} m ({shift:+.1f} m from the raceline apex), {best[0]:.2f} m from the inside wall')
        for kind in ('late', 'early'):
            co, d = res[kind]
            if co.violation(d).max() > 0.05:
                print(f'  note: the {kind} line is still {co.violation(d).max():.2f} m short of where it was asked to be')
        # the two cars together: the attacker (late) starts `gap` m behind the defender (early)
        (te, we), (tl, wl) = times['early apex'], times['late apex']
        delay = args.gap / max(wl[seg[0], 3], 0.1)
        tgrid = np.arange(delay, te[-1], 0.02)
        ie = seg[np.clip(np.searchsorted(te, tgrid), 0, len(seg) - 1)]
        il = seg[np.clip(np.searchsorted(tl + delay, tgrid), 0, len(seg) - 1)]
        dist = np.hypot(*(wl[il, :2] - we[ie, :2]).T)
        behind = (tl[-1] + delay - te[-1]) * wl[seg[-1], 3]   # > 0: the attacker is still behind
        j = int(dist.argmin())
        print(f'  attacker on the late line, {args.gap} m behind a defender on the early line at s = {i0 * ds:.1f} m: '
              f'closest {dist.min():.2f} m at s = {il[j] * ds:.1f} m; at s = {i1 * ds:.1f} m it is '
              f'{abs(behind):.2f} m {"behind" if behind > 0 else "AHEAD"} (gained {args.gap - behind:+.2f} m)')
        li = res['late'][0].idx
        eo = res['early'][0]
        u_in = inside * res['early'][1][eo.idx]
        i_in = apex - eo.before / ds + (int(np.argmax(u_in > 0.15)) if (u_in > 0.15).any() else eo.before / ds)
        corners.append(dict(name=name, inside=inside, s_apex=wrap(apex * ds),
                            s_split=wrap(li[0] * ds),                     # late line = raceline up to here
                            # Decide from where a car `gap` m ahead, on the defender's line, is 0.15 m
                            # inside of the raceline, until the late line starts to move out. (It
                            # used to be only the last 2.3 m of that. A car that covers the inside
                            # brakes early for its tight line, and the ego, still following it
                            # then, braked with it: the later braking point on the outside is
                            # half of what the switchback gains.)
                            s_decide=[wrap(min(i_in * ds - args.gap, apex * ds - args.ramp_from - 2.0)),
                                      wrap(apex * ds - args.ramp_from + 0.3)],
                            s_commit=wrap(apex * ds - 1.0), s_end=wrap(li[-1] * ds),
                            s_defend=wrap(i0 * ds),                       # early line leaves the raceline here
                            s_defend_end=wrap(res['early'][0].idx[-1] * ds),   # ... and is back on it here
                            # the model's prediction for equal cars: what each line costs [s], and
                            # what the attacker gains on the defender through the corner [m]
                            t_raceline=round(float(times['raceline'][0][-1]), 2),
                            t_defend=round(float(te[-1]), 2), t_attack=round(float(tl[-1]), 2),
                            gain_m=round(float(args.gap - behind), 2),
                            v_apex_defend=round(float(we[seg, 3].min()), 2), v_apex_attack=round(float(wl[seg, 3].min()), 2)))
        zones.append((seg, times, delay))
        if c > 0 and (seg[0] - zones[-2][0][-1]) % n > n // 2:
            print(f'  WARNING: the lines of {name} leave the raceline before those of {corners[-2]["name"]} are back on it: '
                  f'the offsets add up there (shorter --early-before / --early-after, or longer straights)')

    base = pathlib.Path(args.raceline).with_suffix('')
    limits = dict(a_lat=args.a_lat, a_lon=args.a_lon, v_min=args.v_min, v_max=args.v_max)
    pathlib.Path(str(base) + '_apex.json').write_text(json.dumps(dict(
        raceline=pathlib.Path(args.raceline).name, n=n, ds=ds, limits=limits, apex_room=args.apex_room,
        entry_inside=args.entry_inside, corners=corners,
        attack=[round(float(x), 4) for x in d_late], defend=[round(float(x), 4) for x in d_early])))
    print()
    for kind, d, what in (('lateapex', d_late, f'{args.apex_room} m outside of the raceline on the way in, then a late apex'),
                          ('earlyapex', d_early, f'{args.entry_inside} m inside of the raceline on the way in')):
        w = lines.offset_line(xy, d)
        t, v = prob.lap(w)
        with open(f'{base}_{kind}.csv', 'w') as f:
            f.write('# x_m,y_m,yaw_rad,v_mps\n')
            f.write(f'# {kind} in every corner of {pathlib.Path(args.raceline).name} (scripts/make_apex_lines.py): {what}\n')
            f.writelines(f'{x:.4f},{y:.4f},{a:.4f},{s:.4f}\n' for (x, y), a, s in zip(w, lines.heading(w), v))
        print(f'wrote {base}_{kind}.csv: whole lap {t:.2f} s (raceline {t_base:.2f} s)')
    print(f'wrote {base}_apex.json')

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    rr, cc = np.nonzero(prob.dist > 0)
    x0, x1 = prob.org[0] + cc.min() * prob.res - 0.3, prob.org[0] + cc.max() * prob.res + 0.3
    y0, y1 = prob.org[1] + rr.min() * prob.res - 0.3, prob.org[1] + rr.max() * prob.res + 0.3
    k = 110
    img = Image.new('RGB', (int((x1 - x0) * k), int((y1 - y0) * k)), (150, 150, 150))
    dr = ImageDraw.Draw(img)
    P = lambda x, y: ((x - x0) * k, (y1 - y) * k)
    for r, c in zip(rr, cc):
        x, y = prob.org[0] + c * prob.res, prob.org[1] + r * prob.res
        dr.rectangle([P(x, y + prob.res), P(x + prob.res, y)], fill=(255, 255, 255))
    pts = [P(*p) for p in xy]
    dr.line(pts + pts[:1], fill=(0, 0, 0), width=3)
    ORANGE, BLUE = (240, 120, 0), (0, 90, 230)
    for co, (seg, times, delay) in zip(corners, zones):
        for label, col, shift in (('early apex', ORANGE, 0.0), ('late apex', BLUE, delay)):
            tt, w = times[label]
            dr.line([P(*w[i, :2]) for i in seg], fill=col, width=4)
            for j, tm in enumerate(np.arange(0.0, tt[-1] + shift, 0.5)):
                if tm < shift:
                    continue
                p = P(*w[seg[int(np.clip(np.searchsorted(tt + shift, tm), 0, len(seg) - 1))], :2])
                dr.ellipse([p[0] - 10, p[1] - 10, p[0] + 10, p[1] + 10], fill=col, outline=(255, 255, 255))
                dr.text((p[0] - 3, p[1] - 5), str(j), fill=(255, 255, 255))
        p = P(*xy[int(co['s_apex'] / ds)])
        dr.ellipse([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], fill=(230, 0, 0))
        dr.text((p[0] + 12, p[1] + 8), co['name'], fill=(230, 0, 0))
    dr.rectangle([10, 10, 470, 100], fill=(255, 255, 255), outline=(0, 0, 0))
    for j, (col, text) in enumerate((((0, 0, 0), 'raceline'), (ORANGE, 'early apex (defender, covers the inside)'),
                                     (BLUE, f'late apex (attacker, starts {args.gap} m behind)'))):
        dr.line([20, 25 + 22 * j, 60, 25 + 22 * j], fill=col, width=4)
        dr.text((70, 19 + 22 * j), text, fill=(0, 0, 0))
    dr.text((20, 82), 'numbered dots: both cars every 0.5 s; red dot: apex of the raceline', fill=(0, 0, 0))
    img.save(str(base) + '_apex.png')
    print(f'wrote {base}_apex.png')


if __name__ == '__main__':
    main()
