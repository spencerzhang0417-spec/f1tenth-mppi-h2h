#!/usr/bin/env python3
"""Switchback test in the simulator: one corner at a time, two equal cars.

    scripts/sb_test.py sbt_a                                # every corner, both cases, 4 / 5 / 6 m/s
    scripts/sb_test.py sbt_a --speeds 6 --corners tight_mid --opp defend --repeat 3
    scripts/sb_test.py sbt_b --out /tmp/sb --ego-params '{"h2h_sb_inside": 0.3}'

Needs the workspace sourced and the venv active (ros2, jax), and the track set up with
scripts/sb_setup.sh. Per corner two cases, each its own short run of the simulator:

    defend    the car ahead covers the inside of this corner (the early-apex line): the switchback
              has to start, and the ego should come out of the corner in front
    raceline  the car ahead stays on the raceline: the switchback must NOT start

Both cars run the same MPPI at the same pace through the corner and after it. On the straight
before it the car ahead is slower (`--opp-straight`, 0.85 of its speed): two equal cars at top
speed never close a gap, and the situation under test is the one a switchback is for, the ego
right behind a car it cannot get past on the straight because that car covers the inside. They
start standing, the ego `--gap` m behind and `--approach` m before the stretch where the
switchback is decided. Only the corner under test is armed, and the ordinary pass on
the straights is off: what gets the ego in front is the switchback and nothing else. The run ends
`--tail` m after the corner's lines are together again.

Measured from the simulator's ground truth (not from what the ego believes): the gap along the
raceline at the decision, at the end of the corner and at the end of the run, the closest the two
cars came, contacts. From the ego's own log: whether and where the switchback started.
"""
import argparse
import csv
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'mppi' / 'scripts'))
import lines  # noqa: E402

CAR_LENGTH = 0.58


def run_case(track, corner, opp, args, tag, extra):
    lset = args.lset   # the line set: <track>_v<top speed>
    import rclpy
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Bool

    wp = HERE.parent / 'mppi' / 'waypoints'
    xy = lines.resample_closed(np.loadtxt(wp / f'{lset}_raceline.csv', delimiter=',', comments='#', ndmin=2)[:, :2])
    L = len(xy) * 0.1
    wrap = lambda d: (d + 0.5 * L) % L - 0.5 * L
    start_s = (corner['s_decide'][0] - args.approach) % L
    run_len = (corner['s_end'] - start_s) % L + args.tail          # the ego drives this far
    log_csv = args.out / f'{tag}.csv'
    cmd = ['ros2', 'launch', 'mppi', 'h2h_sim_launch.py', f'map:={track}', f'raceline:={lset}_raceline.csv',
           'pace:=race', 'latency:=false', 'opp_controller:=mppi', f'switchback:={lset}_raceline_apex.json',
           f'sb_corners:={corner["name"]}', 'h2h_pass:=false', f'start_s:={start_s:.2f}', f'start_gap:={args.gap}',
           f'sync_start:={args.sync}', f'opp_start_delay:={args.opp_delay}', f'h2h_log:={log_csv}']
    if args.opp_straight < 1.0:
        # until where a car on the defender's line starts to brake for the corner
        cmd.append(f'opp_slow_zone:={start_s:.1f},{(corner["s_apex"] - 9.0) % L:.1f},{args.opp_straight}')
    # both cars at the pace the lines were made for
    lim = args.limits
    cmd.append('car_params:=' + json.dumps(dict(a_lat=lim['a_lat'], a_lon=lim['a_lon'], v_min=lim['v_min'],
                                                 v_max=lim['v_max'], max_speed=lim['v_max'] + 0.5)))
    if opp == 'defend':
        cmd += [f'opp_line_set:={lset}_raceline_apex.json', f'opp_line_corners:={corner["name"]}']
    cmd += extra
    launch_log = open(args.out / f'{tag}.log', 'w')
    proc = subprocess.Popen(cmd, stdout=launch_log, stderr=subprocess.STDOUT, start_new_session=True)

    rclpy.init()
    node = rclpy.create_node('sb_test')
    st = {n: dict(xy=None, v=0.0, s=None, total=0.0, coll=0, in_coll=False) for n in ('ego_racecar', 'opp_racecar')}
    series, contacts = [], []

    def odom(msg, name):
        c = st[name]
        p = msg.pose.pose.position
        s = 0.1 * int(np.argmin(np.hypot(xy[:, 0] - p.x, xy[:, 1] - p.y)))
        if c['s'] is not None:
            c['total'] += wrap(s - c['s'])
        c['s'], c['xy'], c['v'] = s, (p.x, p.y), msg.twist.twist.linear.x
        e, o = st['ego_racecar'], st['opp_racecar']
        if name == 'ego_racecar' and o['xy'] is not None:
            series.append((time.monotonic(), e['s'], e['total'], float(wrap(o['s'] - e['s'])), e['v'], o['v'],
                           float(np.hypot(e['xy'][0] - o['xy'][0], e['xy'][1] - o['xy'][1])), *e['xy'], *o['xy']))

    def coll(msg, name):
        c = st[name]
        if msg.data and not c['in_coll']:
            c['coll'] += 1
            contacts.append(dict(car=name, s=c['s'], v=round(c['v'], 2)))
        c['in_coll'] = bool(msg.data)

    for name in st:
        node.create_subscription(Odometry, f'/{name}/odom', lambda m, n=name: odom(m, n), 10)
        node.create_subscription(Bool, f'/{name}/collision', lambda m, n=name: coll(m, n), 10)
    t0 = time.monotonic()
    timeout = args.sync + 20.0 + run_len / 2.0
    while time.monotonic() - t0 < timeout and st['ego_racecar']['total'] < run_len and proc.poll() is None:
        rclpy.spin_once(node, timeout_sec=0.05)
    done = st['ego_racecar']['total'] >= run_len
    node.destroy_node()
    rclpy.try_shutdown()
    try:
        os.killpg(proc.pid, signal.SIGINT)
        for _ in range(30):
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for _ in range(50):      # nothing of this run may be alive when the next one starts
        try:
            os.killpg(proc.pid, 0)
            time.sleep(0.2)
        except ProcessLookupError:
            break
    launch_log.close()

    a = np.array(series) if series else np.zeros((0, 11))
    # t, ego station, ego distance driven, gap, v ego, v opp, distance, x/y ego, x/y opp
    np.save(args.out / f'{tag}.npy', a)
    res = dict(track=track, corner=corner['name'], opp=opp, finished=bool(done), contacts=contacts,
               ego_collisions=st['ego_racecar']['coll'], opp_collisions=st['opp_racecar']['coll'])
    if len(a):
        prog = a[:, 2]

        def at(dist):   # the row where the ego has driven `dist` m
            i = int(np.searchsorted(prog, dist))
            return a[min(i, len(a) - 1)]
        d_dec = (corner['s_decide'][0] - start_s) % L
        d_end = (corner['s_end'] - start_s) % L
        res.update(gap_decide=round(float(at(d_dec)[3]), 2), v_decide=round(float(at(d_dec)[4]), 2),
                   v_opp_decide=round(float(at(d_dec)[5]), 2),
                   gap_corner_end=round(float(at(d_end)[3]), 2), gap_final=round(float(a[-1, 3]), 2),
                   min_dist=round(float(a[:, 6].min()), 2),
                   # the rule book's overtake: half a car length in front
                   overtook=bool(a[-1, 3] < -0.5 * CAR_LENGTH), run_time=round(float(a[-1, 0] - a[0, 0]), 1))
    # the ego's own account
    states, s_first, s_last, n_rows = [], None, None, 0
    if log_csv.exists():
        with open(log_csv) as f:
            for row in csv.DictReader(f):
                n_rows += 1
                if not states or states[-1] != row['state']:
                    states.append(row['state'])
                if row['state'] == 'SWITCHBACK':
                    s_first = float(row['s_e']) if s_first is None else s_first
                    s_last = float(row['s_e'])
    events = [ln.split('h2h -> ', 1)[1].strip() for ln in open(args.out / f'{tag}.log', errors='replace') if 'h2h -> ' in ln]
    res.update(triggered=s_first is not None, sb_from=s_first, sb_to=s_last, states=states, events=events,
               given_up=any('given up' in e and 'switchback' in e for e in events), log_rows=n_rows)
    return res


def verdict(r):
    if not r['finished'] or 'gap_final' not in r:
        return 'NO RUN'
    clean = r['ego_collisions'] == 0 and r['opp_collisions'] == 0
    if r['opp'] == 'raceline' or not r['armed']:
        # nothing to cut under, or a corner where the line model says going wide does not pay
        return ('ok' if r['armed'] else 'ok (not armed)') if (not r['triggered'] and clean) else 'FAIL'
    if not r['triggered']:
        return 'FAIL (no trigger)'
    if not clean:
        return 'FAIL (contact)'
    return 'ok' if r['overtook'] else 'not past'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('track')
    ap.add_argument('--corners', default='', help='names, comma separated (default: all)')
    ap.add_argument('--opp', default='defend,raceline')
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--gap', type=float, default=3.0, help='[m] the ego starts this far behind (center to center)')
    ap.add_argument('--opp-delay', type=float, default=0.8, help='[s] the car ahead sets off this much later')
    ap.add_argument('--opp-straight', type=float, default=0.85,
                    help="the car ahead drives the approach straight at this fraction of its speed (1 = equal everywhere)")
    ap.add_argument('--approach', type=float, default=13.0,
                    help='[m] start this far before the decision stretch (+ 3 m per m/s of top speed over 4)')
    ap.add_argument('--tail', type=float, default=8.0, help='[m] run on this far after the corner')
    ap.add_argument('--speeds', default='4,5,6', help='top speeds [m/s]: the line sets <track>_v<speed> of sb_setup.sh')
    ap.add_argument('--sync', type=float, default=12.0, help='[s] both cars set off this long after the launch')
    ap.add_argument('--min-gain', type=float, default=-9.0,
                    help="[m] the node's h2h_sb_min_gain: corners predicted to gain less are not armed")
    ap.add_argument('--out', type=pathlib.Path, default=pathlib.Path('/tmp/sb_test'))
    ap.add_argument('--ego-params', default='{}', help='json: more parameters for the ego node, e.g. {"h2h_sb_inside": 0.3}')
    ap.add_argument('--launch', action='append', default=[], help='one more launch argument, e.g. --launch latency:=true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ['ROS_DOMAIN_ID'] = os.environ.get('H2H_DOMAIN_ID', '77')   # not the simulator the user may have open

    only = [c for c in args.corners.split(',') if c]
    results, approach = [], args.approach
    for speed in args.speeds.split(','):
        args.lset = f'{args.track}_v{speed}'
        spec = json.loads((HERE.parent / 'mppi' / 'waypoints' / f'{args.lset}_raceline_apex.json').read_text())
        args.limits = spec['limits']
        args.approach = approach + 3.0 * max(spec['limits']['v_max'] - 4.0, 0.0)   # room to get up to speed
        print(f'{args.lset}: both cars at a_lat {spec["limits"]["a_lat"]}, a_lon {spec["limits"]["a_lon"]} m/s^2, '
              f'v_max {spec["limits"]["v_max"]} m/s', flush=True)
        for corner in spec['corners']:
            if only and corner['name'] not in only:
                continue
            for opp in args.opp.split(','):
                for k in range(args.repeat):
                    tag = f'{args.lset}_{corner["name"]}_{opp}_{k + 1}'
                    extra = args.launch + ['ego_params:=' + json.dumps({**json.loads(args.ego_params),
                                                                        'h2h_sb_min_gain': float(args.min_gain)})]
                    r = run_case(args.track, corner, opp, args, tag, extra)
                    r['v_max'] = spec['limits']['v_max']
                    r['tag'] = tag
                    r['gain_model'] = corner.get('gain_m')
                    r['armed'] = corner.get('gain_m', 9.9) >= args.min_gain
                    r['verdict'] = verdict(r)
                    results.append(r)
                    print(f'{tag:38s} {r["verdict"]:18s} trigger {str(r["triggered"]):5s} '
                          + (f'gap {r["gap_decide"]:+.2f} -> {r["gap_corner_end"]:+.2f} -> {r["gap_final"]:+.2f} m, '
                             f'closest {r["min_dist"]:.2f} m, contacts {r["ego_collisions"]}/{r["opp_collisions"]}'
                             if 'gap_final' in r else 'no data'), flush=True)
                    (args.out / f'{args.track}_results.json').write_text(json.dumps(results, indent=1))
    print(f'\n(gap = opponent minus ego along the raceline: + = the ego is behind; at the decision -> where the '
          f'lines rejoin -> {args.tail:.0f} m later)\nresults: {args.out / (args.track + "_results.json")}')


if __name__ == '__main__':
    main()
