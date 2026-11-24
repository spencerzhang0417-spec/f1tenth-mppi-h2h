#!/usr/bin/env python3
"""Picture of one run of scripts/sb_test.py: what the two cars really drove through the corner.

    scripts/sb_plot.py /tmp/sb_test/sbt_a_tight_mid_defend_1.npy [more.npy ...]

Writes <run>.png next to each: the map around the corner, the raceline (grey), the defender's
and the attacker's planned lines (thin orange / blue), and the driven paths (thick), with both
cars drawn every 0.5 s; equal numbers are the same instant.
"""
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'mppi' / 'scripts'))
sys.path.insert(0, str(HERE))
import lines  # noqa: E402
from make_track_bounds import load_map  # noqa: E402

ORANGE, BLUE = (235, 120, 0), (0, 90, 230)


def car(dr, P, x, y, yaw, col, label):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = [P(x + c * a - s * b, y + s * a + c * b) for a, b in ((0.29, 0.155), (0.29, -0.155), (-0.29, -0.155), (-0.29, 0.155))]
    dr.polygon(pts, fill=col, outline=(255, 255, 255))
    p = P(x, y)
    dr.text((p[0] - 3, p[1] - 6), label, fill=(255, 255, 255))


def plot(npy):
    npy = pathlib.Path(npy)
    a = np.load(npy)
    track = '_'.join(npy.stem.split('_')[:2])
    lset = '_'.join(npy.stem.split('_')[:3])      # <track>_v<top speed>
    root = HERE.parent / 'mppi'
    free, res, org = load_map(root / 'maps' / f'{track}.yaml')
    xy = lines.resample_closed(np.loadtxt(root / 'waypoints' / f'{lset}_raceline.csv', delimiter=',', comments='#', ndmin=2)[:, :2])
    ls = json.loads((root / 'waypoints' / f'{lset}_raceline_apex.json').read_text())
    corner = next((c for c in ls['corners'] if f'_{c["name"]}_' in npy.stem), None)
    a = a[a[:, 4] > 0.3] if (a[:, 4] > 0.3).any() else a           # from when the ego moves
    pts = np.vstack([a[:, 7:9], a[:, 9:11]])
    x0, x1, y0, y1 = pts[:, 0].min() - 1.2, pts[:, 0].max() + 1.2, pts[:, 1].min() - 1.2, pts[:, 1].max() + 1.2
    k = min(1500 / (x1 - x0), 1100 / (y1 - y0))
    img = Image.new('RGB', (int((x1 - x0) * k), int((y1 - y0) * k)), (120, 120, 120))
    dr = ImageDraw.Draw(img)
    P = lambda x, y: ((x - x0) * k, (y1 - y) * k)
    rr, cc = np.nonzero(free)
    for r, c in zip(rr, cc):
        x, y = org[0] + c * res, org[1] + r * res
        if x0 - res < x < x1 and y0 - res < y < y1:
            dr.rectangle([P(x, y + res), P(x + res, y)], fill=(250, 250, 250))

    def line(p, col, width):
        q = [P(*v) for v in p]
        for u, v in zip(q, q[1:] + q[:1]):
            if abs(u[0] - v[0]) + abs(u[1] - v[1]) < 60:
                dr.line([u, v], fill=col, width=width)
    line(xy, (170, 170, 170), 2)
    if corner is not None:
        n = len(xy)
        end = max(corner['s_end'], corner.get('s_defend_end', corner['s_end']), key=lambda q: (q - corner['s_apex']) % (n * 0.1))
        zone = (int(corner['s_defend'] / 0.1) + np.arange(int(((end - corner['s_defend']) % (n * 0.1)) / 0.1) + 1)) % n
        for key, col in (('defend', ORANGE), ('attack', BLUE)):
            w = lines.offset_line(xy, lines.fit_offsets(ls[key], n))[zone]
            dr.line([P(*v) for v in w], fill=tuple(int(0.5 * c + 127) for c in col), width=2)
    dr.line([P(*v) for v in a[:, 9:11]], fill=ORANGE, width=4)
    dr.line([P(*v) for v in a[:, 7:9]], fill=BLUE, width=4)
    t = a[:, 0] - a[0, 0]
    for j, tm in enumerate(np.arange(0.0, t[-1], 0.5)):
        i = int(np.searchsorted(t, tm))
        i2 = min(i + 3, len(a) - 1)
        for cols, col in ((slice(9, 11), ORANGE), (slice(7, 9), BLUE)):
            d = a[i2, cols] - a[max(i - 3, 0), cols]
            car(dr, P, a[i, cols][0], a[i, cols][1], float(np.arctan2(d[1], d[0])), col, str(j))
    res_json = npy.parent / f'{track}_results.json'
    head = npy.stem
    if res_json.exists():
        for r in json.loads(res_json.read_text()):
            if r.get('tag', f'{lset}_{r["corner"]}_{r["opp"]}_1') == npy.stem and 'gap_final' in r:
                head = (f'{npy.stem}: {r.get("verdict", "")}, switchback {"started" if r["triggered"] else "not started"}, '
                        f'gap {r["gap_decide"]:+.2f} -> {r["gap_corner_end"]:+.2f} -> {r["gap_final"]:+.2f} m, closest {r["min_dist"]:.2f} m')
    dr.rectangle([0, 0, img.size[0], 34], fill=(255, 255, 255))
    dr.text((8, 4), head, fill=(0, 0, 0))
    dr.text((8, 18), 'orange: car ahead (thin: its planned line)   blue: ego (thin: late-apex line)   grey: raceline   '
                     'cars every 0.5 s, equal numbers = same instant', fill=(0, 0, 0))
    out = npy.with_suffix('.png')
    img.save(out)
    print(f'wrote {out}')


if __name__ == '__main__':
    for f in sys.argv[1:]:
        plot(f)
