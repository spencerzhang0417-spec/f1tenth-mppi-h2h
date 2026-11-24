#!/usr/bin/env python3
"""Test tracks for the switchback: closed loops whose corners differ in inner radius and width.

    python3 scripts/make_test_track.py            # all tracks below
    python3 scripts/make_test_track.py sbt_a      # one of them

A track is a closed polygon driven counter-clockwise (left-hand corners: the car steers further
to the left than to the right). Every vertex is one corner, given by what a driver sees of it:

    r_in   radius of the INSIDE wall [m] (how tight the corner is)
    width  wall to wall [m], held through the corner and the first / last part of the straights
           next to it; half way down a straight it blends into the width of the next corner

The centerline is the polygon with every vertex rounded with r_in + width / 2. Two 90 degree
vertices with no straight between them are one 180 degree hairpin; a vertex with an acute
inside angle is an acute corner (the track turns by more than 90 degrees there).

Written per track <name>:
    mppi/maps/<name>.pgm / .yaml        the map (0.05 m per cell), free inside the walls
    mppi/waypoints/<name>_center.csv    the centerline every 0.1 m: x_m, y_m
    mppi/waypoints/<name>_corners.json  the corners: name, r_in, width, turn angle, station of
                                        their middle on the centerline
    mppi/maps/<name>.png                picture with the corner names

Then: scripts/optimize_raceline.py for the raceline, scripts/make_apex_lines.py for the
attacker's and the defender's lines (scripts/sb_setup.sh does all of it).
"""
import json
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = pathlib.Path(__file__).resolve().parent
RES = 0.05

# name: [(x, y, r_in, width, corner name), ...] counter-clockwise
TRACKS = {
    # four 90 degree corners, straights of 21 m between them: long enough that the lines of one
    # corner are back on the raceline before those of the next one leave it, and to be at 6 m/s
    # again before the next braking point
    'sbt_a': [
        (25.0, 0.0, 0.6, 3.0, 'tight_wide'),
        (25.0, 25.0, 0.6, 2.2, 'tight_mid'),
        (0.0, 25.0, 1.5, 2.2, 'open_mid'),
        (0.0, 0.0, 0.6, 1.6, 'tight_narrow'),
    ],
    # two 180 degree hairpins (each two 90 degree vertices 2 x 2.1 m apart), 20 m straights
    'sbt_b': [
        (22.1, 0.0, 0.6, 3.0, 'hairpin_wide'), (22.1, 4.2, 0.6, 3.0, None),
        (-2.1, 4.2, 1.0, 2.2, 'hairpin_mid'), (-2.1, 0.0, 1.0, 2.2, None),
    ],
    # a right triangle: two acute corners (the track turns 135 degrees) and a 90 degree one
    'sbt_c': [
        (28.0, 0.0, 0.6, 3.0, 'acute_wide'),
        (0.0, 28.0, 0.6, 2.2, 'acute_mid'),
        (0.0, 0.0, 1.0, 2.2, 'square_mid'),
    ],
}


def centerline(verts, step=0.02):
    """the rounded polygon: points every ~step m, and per point the index of the corner whose
    arc it is on (-1 on a straight)"""
    n = len(verts)
    p = np.array([v[:2] for v in verts], dtype=float)
    r = np.array([v[2] + 0.5 * v[3] for v in verts])
    pts, tag = [], []
    for k in range(n):
        a, b, c = p[k - 1], p[k], p[(k + 1) % n]
        u, w = (b - a) / np.linalg.norm(b - a), (c - b) / np.linalg.norm(c - b)
        turn = np.arctan2(u[0] * w[1] - u[1] * w[0], u @ w)         # + = left
        t = r[k] * np.tan(abs(turn) / 2.0)                          # corner start / end to the vertex
        # the straight from the end of the previous arc to the start of this one
        r_prev = r[k - 1]
        up = (a - p[k - 2]) / np.linalg.norm(a - p[k - 2])
        turn_prev = np.arctan2(up[0] * u[1] - up[1] * u[0], up @ u)
        t_prev = r_prev * np.tan(abs(turn_prev) / 2.0)
        length = np.linalg.norm(b - a) - t_prev - t
        if length < -1e-6:
            raise ValueError(f'corners {k - 1} and {k} overlap by {-length:.2f} m')
        m = max(int(round(length / step)), 0)
        for j in range(m):
            pts.append(a + u * (t_prev + length * j / max(m, 1)))
            tag.append(-1)
        # the arc
        side = np.sign(turn)
        center = b - u * t + side * r[k] * np.array([-u[1], u[0]])
        a0 = np.arctan2(*(b - u * t - center)[::-1])
        m = max(int(round(r[k] * abs(turn) / step)), 1)
        for j in range(m):
            ang = a0 + turn * j / m
            pts.append(center + r[k] * np.array([np.cos(ang), np.sin(ang)]))
            tag.append(k)
    return np.array(pts), np.array(tag)


def turn_deg(verts, k):
    """how far the track turns at vertex k [deg, + = left]"""
    p = np.array([v[:2] for v in verts], dtype=float)
    u, w = p[k] - p[k - 1], p[(k + 1) % len(p)] - p[k]
    return float(np.degrees(np.arctan2(u[0] * w[1] - u[1] * w[0], u @ w)))


def widths(verts, xy, tag, hold=0.35):
    """width at every centerline point: the corner's own through its arc and the nearest `hold`
    part of each straight next to it, blended over the middle of the straight"""
    n, m = len(verts), len(xy)
    seg = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)
    w = np.zeros(m)
    i = 0
    while i < m:
        if tag[i] >= 0:
            w[i] = verts[tag[i]][3]
            i += 1
            continue
        j = i
        while j < m and tag[j] < 0:
            j += 1
        k_next = tag[j % m]
        k_prev = tag[i - 1]
        s = np.r_[0.0, np.cumsum(seg[i:j])][:j - i] / max(float(np.sum(seg[i:j])), 1e-6)
        t = np.clip((s - hold) / (1.0 - 2.0 * hold), 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        w[i:j] = (1.0 - t) * verts[k_prev][3] + t * verts[k_next][3]
        i = j
    return w


def build(name, verts, out):
    xy, tag = centerline(verts)
    w = widths(verts, xy, tag)
    margin = 0.5 * max(v[3] for v in verts) + 0.6
    x0, y0 = xy[:, 0].min() - margin, xy[:, 1].min() - margin
    cols = int(np.ceil((xy[:, 0].max() + margin - x0) / RES))
    rows = int(np.ceil((xy[:, 1].max() + margin - y0) / RES))
    cx = x0 + (np.arange(cols) + 0.5) * RES
    cy = y0 + (np.arange(rows) + 0.5) * RES
    gx, gy = np.meshgrid(cx, cy)
    dist, idx = cKDTree(xy).query(np.c_[gx.ravel(), gy.ravel()])
    free = (dist <= 0.5 * w[idx]).reshape(rows, cols)        # row 0 = bottom

    from PIL import Image, ImageDraw
    maps, wps = out / 'maps', out / 'waypoints'
    Image.fromarray(np.where(free[::-1], 254, 0).astype(np.uint8)).save(maps / f'{name}.pgm')
    (maps / f'{name}.yaml').write_text(
        f'image: {name}.pgm\nmode: trinary\nresolution: {RES}\norigin:\n  - {x0:.3f}\n  - {y0:.3f}\n  - 0\n'
        'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')

    # the centerline every 0.1 m, and where the corners are on it
    seg = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)
    s = np.r_[0.0, np.cumsum(seg)[:-1]]
    length = float(seg.sum())
    m = int(length / 0.1)
    s_new = np.linspace(0.0, length, m, endpoint=False)
    xc = np.c_[np.interp(s_new, np.r_[s, length], np.r_[xy[:, 0], xy[0, 0]]),
               np.interp(s_new, np.r_[s, length], np.r_[xy[:, 1], xy[0, 1]])]
    with open(wps / f'{name}_center.csv', 'w') as f:
        f.write(f'# x_m,y_m\n# centerline of the test track {name} (scripts/make_test_track.py)\n')
        f.writelines(f'{x:.4f},{y:.4f}\n' for x, y in xc)
    corners, k = [], 0
    while k < len(verts):
        ks = [k]
        if verts[k][4] is not None and k + 1 < len(verts) and verts[k + 1][4] is None:
            ks.append(k + 1)                                  # a hairpin: two vertices
        on = np.isin(tag, ks)
        corners.append(dict(name=verts[k][4], r_in=verts[k][2], width=verts[k][3],
                            turn_deg=int(round(sum(turn_deg(verts, q) for q in ks))),
                            s_from=round(float(s[on].min()), 2), s_to=round(float(s[on].max()), 2),
                            s_mid=round(float(0.5 * (s[on].min() + s[on].max())), 2),
                            xy_mid=[round(float(v), 3) for v in xy[np.nonzero(on)[0][on.sum() // 2]]]))
        k += len(ks)
    (wps / f'{name}_corners.json').write_text(json.dumps(dict(track=name, length=round(length, 2), corners=corners), indent=1))

    scale = 40
    img = Image.fromarray(np.where(free[::-1], 255, 90).astype(np.uint8)).convert('RGB').resize((cols * 2, rows * 2))
    dr = ImageDraw.Draw(img)
    P = lambda x, y: ((x - x0) * scale, (rows * RES - (y - y0)) * scale)
    dr.line([P(*q) for q in xc] + [P(*xc[0])], fill=(160, 160, 160), width=1)
    for c in corners:
        q = xc[int(c['s_mid'] / 0.1) % m]
        ctr = xc.mean(axis=0)
        d = (ctr - q) / np.linalg.norm(ctr - q)
        px = P(*(q + d * (0.5 * c['width'] + 0.9)))
        dr.text((px[0] - 50, px[1] - 12), f'{c["name"]}\nr_in {c["r_in"]} m, width {c["width"]} m', fill=(255, 220, 0))
    img.save(maps / f'{name}.png')
    print(f'{name}: lap {length:.1f} m on the centerline, map {cols} x {rows} cells; corners: '
          + ', '.join(f'{c["name"]} ({c["turn_deg"]} deg, r_in {c["r_in"]}, width {c["width"]}, s = {c["s_mid"]:.1f} m)'
                      for c in corners))


def main():
    out = HERE.parent / 'mppi'
    for name in (sys.argv[1:] or list(TRACKS)):
        build(name, TRACKS[name], out)


if __name__ == '__main__':
    main()
