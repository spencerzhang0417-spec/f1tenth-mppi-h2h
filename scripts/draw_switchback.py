#!/usr/bin/env python3
"""Picture of the switchback through one corner: both cars at equally spaced instants.

    python3 scripts/draw_switchback.py                       # corner 2 of race2_raceline_opt
    python3 scripts/draw_switchback.py --corner corner1 --gap 1.5 --defender-a-lat 4.3

The lines are the early-apex (defender) and late-apex (attacker) lines of
scripts/make_apex_lines.py. The cars are placed by the time each needs along its line with the
speed profile of the lap-time model (perfect tracking). --defender-a-lat sets the lateral
acceleration the DEFENDER's profile is built with: 4.3 = the same car as the attacker; the
default 2.77 makes its slowest point 1.7 m/s, what the early-apex opponent did in the simulator.
"""
import argparse
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'mppi' / 'scripts'))
sys.path.insert(0, str(HERE))
import lines  # noqa: E402
from make_track_bounds import load_map  # noqa: E402

ORANGE, BLUE = (235, 110, 0), (0, 90, 225)
CAR_L, CAR_W = 0.58, 0.31
CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'   # has the circled digits of the labels
FOOT = 185   # [px] legend and assumptions below the picture


def font(size):
    try:
        return ImageFont.truetype(CJK, size, index=2)
    except OSError:
        return ImageFont.load_default()


def main():
    root = HERE.parent / 'mppi'
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--map', default=str(root / 'maps' / 'race2_map_edited.yaml'))
    ap.add_argument('--lines', default=str(root / 'waypoints' / 'race2_raceline_opt_apex.json'))
    ap.add_argument('--corner', default='corner2')
    ap.add_argument('--gap', type=float, default=1.0, help='[m] attacker behind the defender at the start')
    ap.add_argument('--defender-a-lat', type=float, default=2.77)
    ap.add_argument('--after', type=float, default=8.6, help='[m] past the apex the picture ends')
    ap.add_argument('--instants', type=int, default=6)
    ap.add_argument('--out', default=str(root / 'waypoints' / 'switchback_diagram.png'))
    args = ap.parse_args()

    ls = lines.load_line_set(args.lines)
    xy = lines.resample_closed(np.loadtxt(pathlib.Path(args.lines).parent / ls['raceline'],
                                          delimiter=',', comments='#', ndmin=2)[:, :2])
    n, ds, lim = len(xy), 0.1, ls['limits']
    co = next(c for c in ls['corners'] if c['name'] == args.corner)
    att = lines.make_waypoints(xy, lines.fit_offsets(ls['attack'], n), **lim)
    de = lines.make_waypoints(xy, lines.fit_offsets(ls['defend'], n), **dict(lim, a_lat=args.defender_a_lat))
    i0 = int(co['s_defend'] / ds)
    seg = (i0 + np.arange(int(((co['s_apex'] + args.after - co['s_defend']) % (n * ds)) / ds) + 1)) % n

    def clock(w):
        sl = np.hypot(*(w[seg[1:], :2] - w[seg[:-1], :2]).T)
        return np.r_[0.0, np.cumsum(sl / np.maximum(0.5 * (w[seg[:-1], 3] + w[seg[1:], 3]), 1e-3))]

    ta, td = clock(att) + args.gap / att[seg[0], 3], clock(de)
    t_end = min(ta[-1], td[-1])
    times = np.linspace(0.3, t_end - 0.1, args.instants)
    ia = [int(seg[min(np.searchsorted(ta, t), len(seg) - 1)]) for t in times]
    idf = [int(seg[min(np.searchsorted(td, t), len(seg) - 1)]) for t in times]

    # ---- canvas ----
    free, res, org = load_map(args.map)
    pts = np.vstack([att[seg, :2], de[seg, :2]])
    x0, x1 = pts[:, 0].min() - 1.0, pts[:, 0].max() + 4.2   # room on the right for the labels
    y0, y1 = pts[:, 1].min() - 1.5, pts[:, 1].max() + 1.2
    k = 175
    W, H = int((x1 - x0) * k), int((y1 - y0) * k)
    img = Image.new('RGB', (W, H + FOOT), (255, 255, 255))
    dr = ImageDraw.Draw(img, 'RGBA')
    P = lambda x, y: ((x - x0) * k, (y1 - y) * k)
    rr, cc = np.nonzero(~free)
    for r, c in zip(rr, cc):
        x, y = org[0] + c * res, org[1] + r * res
        if x0 - 0.1 < x < x1 and y0 - 0.1 < y < y1:
            dr.rectangle([P(x, y + res), P(x + res, y)], fill=(120, 120, 120))
    dr.rectangle([0, H, W, H + FOOT], fill=(255, 255, 255))

    # raceline dashed, the two lines solid
    for j in range(0, len(seg) - 2, 4):
        dr.line([P(*xy[seg[j]]), P(*xy[seg[j + 2]])], fill=(90, 90, 90), width=2)
    dr.line([P(*p) for p in de[seg, :2]], fill=ORANGE + (255,), width=5)
    dr.line([P(*p) for p in att[seg, :2]], fill=BLUE + (255,), width=5)

    def car(w, i, col, num, shade):
        x, y, yaw = w[i, 0], w[i, 1], w[i, 2]
        c, s = np.cos(yaw), np.sin(yaw)
        corners = [(x + a * c - b * s, y + a * s + b * c) for a, b in
                   ((CAR_L / 2, CAR_W / 2), (CAR_L / 2, -CAR_W / 2), (-CAR_L / 2, -CAR_W / 2), (-CAR_L / 2, CAR_W / 2))]
        fill = tuple(int(255 - (255 - v) * shade) for v in col)
        dr.polygon([P(*q) for q in corners], fill=fill + (255,), outline=col + (255,))
        nose = P(x + 0.36 * c, y + 0.36 * s)
        dr.line([P(x + 0.29 * c, y + 0.29 * s), nose], fill=col + (255,), width=4)
        px, py = P(x, y)
        f = font(22)
        dr.text((px - 6, py - 15), str(num), fill=(255, 255, 255) if shade > 0.55 else (0, 0, 0), font=f)

    for j in range(args.instants):
        shade = 0.35 + 0.65 * j / max(args.instants - 1, 1)   # later = darker
        car(de, idf[j], ORANGE, j + 1, shade)
        car(att, ia[j], BLUE, j + 1, shade)

    # where the lines cross after the apex: the attacker passes behind the defender here
    da, dd = lines.fit_offsets(ls['attack'], n), lines.fit_offsets(ls['defend'], n)
    ap_i = int(co['s_apex'] / ds)
    after = (ap_i + np.arange(-10, 40)) % n
    cross = [i for i, j in zip(after[:-1], after[1:]) if (da[i] - dd[i]) * (da[j] - dd[j]) <= 0]
    f_small, f_mid, f_big = font(24), font(28), font(36)

    def label(text, at, offset, col=(0, 0, 0)):
        px, py = P(*at)
        tx, ty = px + offset[0], py + offset[1]
        box = dr.multiline_textbbox((tx, ty), text, font=f_small, spacing=4)
        dr.rectangle([box[0] - 8, box[1] - 6, box[2] + 8, box[3] + 6], fill=(255, 255, 255, 235), outline=col + (255,))
        dr.multiline_text((tx, ty), text, fill=col, font=f_small, spacing=4)
        cx = min(max(px, box[0]), box[2])
        cy = min(max(py, box[1]), box[3])
        dr.line([(px, py), (cx, cy)], fill=col + (255,), width=2)

    if cross:
        cx, cy = 0.5 * (att[cross[0], :2] + de[cross[0], :2])
        px, py = P(cx, cy)
        dr.ellipse([px - 16, py - 16, px + 16, py + 16], outline=(0, 0, 0, 255), width=4)
        dr.line([px - 11, py - 11, px + 11, py + 11], fill=(0, 0, 0, 255), width=3)
        dr.line([px - 11, py + 11, px + 11, py - 11], fill=(0, 0, 0, 255), width=3)
        label('Crossing point: the attacker cuts back\nto the inside behind the defender', (cx, cy), (150, 60))
    v = lambda w, i: f'{w[i, 3]:.1f} m/s'
    label('① Entry: the defender covers the inside (early apex);\n    the attacker stays outside and does not follow it in',
          de[idf[1], :2], (-880, 190), ORANGE)
    label(f'② Apex: the defender\'s line is too tight:\n    down to {de[seg, 3].min():.1f} m/s, and it runs wide',
          de[int(seg[np.argmin(de[seg, 3])]), :2], (130, -70), ORANGE)
    label(f'③ The attacker goes deep and turns late,\n    slowest point {att[seg, 3].min():.1f} m/s', att[ia[2], :2], (-60, 120), BLUE)
    label(f'④ Exit: attacker on the inside at {v(att, ia[-2])},\n    defender on the outside at {v(de, idf[-2])}', att[ia[-2], :2], (-330, 60), BLUE)
    lead = (td[-1] - ta[-1]) * att[seg[-1], 3]
    label(f'⑤ Straight: attacker {lead:.1f} m ahead' if lead > 0 else f'⑤ Straight: attacker still {-lead:.1f} m behind',
          att[ia[-1], :2], (-120, -110), BLUE)

    # legend and the assumptions, below the picture
    dr.text((20, H + 10), 'Switchback overtake (%s, %s-hand corner)'
            % (args.corner, 'left' if co['inside'] > 0 else 'right'), fill=(0, 0, 0), font=f_big)
    x = 20
    for text, col in (('Defender: early-apex line', ORANGE), ('Attacker: late-apex line', BLUE), ('raceline', None)):
        if col is None:
            for j in range(0, 60, 16):
                dr.line([x + j, H + 78, x + j + 8, H + 78], fill=(90, 90, 90, 255), width=3)
        else:
            dr.line([x, H + 78, x + 60, H + 78], fill=col + (255,), width=6)
        dr.text((x + 70, H + 60), text, fill=(0, 0, 0), font=f_mid)
        x += 70 + int(dr.textlength(text, font=f_mid)) + 50
    same = 'the defender is the same car' if abs(args.defender_a_lat - lim['a_lat']) < 1e-6 else \
        f"the defender's slowest point is the {de[seg, 3].min():.1f} m/s measured in the simulator"
    dr.multiline_text((20, H + 105),
                      f'Equal numbers = the same instant, {times[1] - times[0]:.2f} s apart; darker = later. '
                      f'The attacker starts {args.gap:.1f} m behind.\n'
                      f'Positions from the speed model (perfect tracking); {same}.',
                      fill=(60, 60, 60), font=f_small, spacing=6)
    img.save(args.out)
    print(f'wrote {args.out} ({img.size[0]} x {img.size[1]}); lead at the end {lead:+.2f} m; '
          f'defender slowest {de[seg, 3].min():.2f}, attacker slowest {att[seg, 3].min():.2f} m/s')
    for j, t in enumerate(times):
        print(f'  {j + 1}: t {t:.2f} s  attacker s {ia[j] * ds:5.1f} v {att[ia[j], 3]:.2f} | defender s {idf[j] * ds:5.1f} v {de[idf[j], 3]:.2f} '
              f'| apart {np.hypot(*(att[ia[j], :2] - de[idf[j], :2])):.2f} m')


if __name__ == '__main__':
    main()
