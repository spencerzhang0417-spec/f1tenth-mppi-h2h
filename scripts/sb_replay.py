#!/usr/bin/env python3
"""Replay the switchback through one corner over and over, to watch it in Foxglove.

    scripts/sb_replay.py                                     # sbt_a tight_wide at 6 m/s
    scripts/sb_replay.py sbt_c acute_wide --speed 6
    scripts/sb_replay.py sbt_a tight_wide --speed 5 --loops 3 --no-foxglove

Needs the workspace sourced and the venv active, and the tracks set up (scripts/sb_setup.sh).
The simulator is started once, with the same two cars and the same situation as
scripts/sb_test.py: the car ahead covers the inside of the corner, the ego starts behind it on
the straight before. Every time the ego is `--tail` m past the corner both cars are put back on
their start poses (the simulator's /initialpose and /goal_pose resets) and set off again. Per pass a line with
the gap before and after the corner; stop with Ctrl-C.

Foxglove: the viewer opens on its own (`--port`, 8766 by default so that a simulator already
open on the usual 8765 is left alone). In the 3D panel switch on /h2h/markers (state text over
the ego, raceline white, late-apex line blue, tracked car red), or import the layout
mppi/config/foxglove_h2h.json.
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('track', nargs='?', default='sbt_a')
    ap.add_argument('corner', nargs='?', default='tight_wide')
    ap.add_argument('--speed', default='6', help='top speed [m/s]: the line set <track>_v<speed>')
    ap.add_argument('--opp', default='defend', choices=('defend', 'raceline'))
    ap.add_argument('--loops', type=int, default=0, help='passes through the corner (0 = until Ctrl-C)')
    ap.add_argument('--gap', type=float, default=3.0)
    ap.add_argument('--opp-delay', type=float, default=0.8)
    ap.add_argument('--opp-straight', type=float, default=0.85)
    ap.add_argument('--approach', type=float, default=13.0)
    ap.add_argument('--tail', type=float, default=8.0)
    ap.add_argument('--pause', type=float, default=2.0, help='[s] the cars stand on their start poses before each pass')
    ap.add_argument('--port', default='8766', help='port of the Foxglove bridge')
    ap.add_argument('--no-foxglove', action='store_true', help='do not open the viewer')
    ap.add_argument('--visualize', action='store_true', help='also publish the MPPI rollouts (/mppi/visualization)')
    args = ap.parse_args()
    os.environ['ROS_DOMAIN_ID'] = os.environ.get('H2H_DOMAIN_ID', '77')

    import rclpy
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Bool, Float64

    wp = HERE.parent / 'mppi' / 'waypoints'
    lset = f'{args.track}_v{args.speed}'
    spec = json.loads((wp / f'{lset}_raceline_apex.json').read_text())
    corner = next(c for c in spec['corners'] if c['name'] == args.corner)
    lim = spec['limits']
    xy = lines.resample_closed(np.loadtxt(wp / f'{lset}_raceline.csv', delimiter=',', comments='#', ndmin=2)[:, :2])
    yaw = lines.heading(xy)
    L = len(xy) * 0.1
    wrap = lambda d: (d + 0.5 * L) % L - 0.5 * L
    start_s = (corner['s_decide'][0] - args.approach - 3.0 * max(lim['v_max'] - 4.0, 0.0)) % L
    run_len = (corner['s_end'] - start_s) % L + args.tail
    sync = 12.0
    cmd = ['ros2', 'launch', 'mppi', 'h2h_sim_launch.py', f'map:={args.track}', f'raceline:={lset}_raceline.csv',
           'pace:=race', 'latency:=false', 'opp_controller:=mppi', f'switchback:={lset}_raceline_apex.json',
           f'sb_corners:={corner["name"]}', 'h2h_pass:=false', f'start_s:={start_s:.2f}', f'start_gap:={args.gap}',
           f'sync_start:={sync}', f'opp_start_delay:={args.opp_delay}',
           f'opp_slow_zone:={start_s:.1f},{(corner["s_apex"] - 9.0) % L:.1f},{args.opp_straight}',
           'car_params:=' + json.dumps(dict(a_lat=lim['a_lat'], a_lon=lim['a_lon'], v_min=lim['v_min'],
                                            v_max=lim['v_max'], max_speed=lim['v_max'] + 0.5)),
           'ego_params:=' + json.dumps({'h2h_sb_min_gain': -9.0}),
           f'open_foxglove:={"false" if args.no_foxglove else "true"}', f'foxglove_port:={args.port}',
           f'visualize:={"true" if args.visualize else "false"}']
    if args.opp == 'defend':
        cmd += [f'opp_line_set:={lset}_raceline_apex.json', f'opp_line_corners:={corner["name"]}']
    log_path = pathlib.Path(os.environ.get('TMPDIR', '/tmp')) / f'sb_replay_{os.getpid()}.log'
    log = open(log_path, 'w')
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f'{lset} {corner["name"]}: both cars at v_max {lim["v_max"]} m/s, the car ahead '
          f'{"covers the inside" if args.opp == "defend" else "stays on the raceline"}; launch log {log_path}')
    if not args.no_foxglove:
        print(f'Foxglove: ws://localhost:{args.port}  (3D panel: switch on /h2h/markers)')

    rclpy.init()
    node = rclpy.create_node('sb_replay')
    st = {n: dict(xy=None, s=None, total=0.0, coll=0, in_coll=False) for n in ('ego_racecar', 'opp_racecar')}
    gaps = []

    def odom(msg, name):
        c = st[name]
        p = msg.pose.pose.position
        s = 0.1 * int(np.argmin(np.hypot(xy[:, 0] - p.x, xy[:, 1] - p.y)))
        if c['s'] is not None and abs(wrap(s - c['s'])) < 2.0:      # (not the jump back to the start)
            c['total'] += wrap(s - c['s'])
        c['s'], c['xy'] = s, (p.x, p.y)
        e, o = st['ego_racecar'], st['opp_racecar']
        if name == 'ego_racecar' and o['s'] is not None:
            gaps.append((e['total'], float(wrap(o['s'] - e['s']))))

    def coll(msg, name):
        c = st[name]
        c['coll'] += bool(msg.data and not c['in_coll'])
        c['in_coll'] = bool(msg.data)

    for name in st:
        node.create_subscription(Odometry, f'/{name}/odom', lambda m, n=name: odom(m, n), 10)
        node.create_subscription(Bool, f'/{name}/collision', lambda m, n=name: coll(m, n), 10)
    reset_pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 1)
    restart_pub = node.create_publisher(Float64, '/sim/restart', 1)
    opp_pub = node.create_publisher(PoseStamped, '/goal_pose', 1)
    i0, i1 = int(start_s / 0.1) % len(xy), int((start_s + args.gap) / 0.1) % len(xy)
    pose, opp_pose = PoseWithCovarianceStamped(), PoseStamped()
    pose.header.frame_id = opp_pose.header.frame_id = 'map'
    for p, i in ((pose.pose.pose, i0), (opp_pose.pose, i1)):
        p.position.x, p.position.y = float(xy[i, 0]), float(xy[i, 1])
        p.orientation.z, p.orientation.w = float(np.sin(0.5 * yaw[i])), float(np.cos(0.5 * yaw[i]))

    def events():
        with open(log_path, errors='replace') as f:
            return [ln.split('h2h -> ', 1)[1].strip() for ln in f if 'h2h -> ' in ln]

    def spin_until(cond, timeout):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout and proc.poll() is None and not cond():
            rclpy.spin_once(node, timeout_sec=0.05)
        return cond()

    k, past, seen = 0, 0, 0
    d_dec = (corner['s_decide'][0] - start_s) % L
    try:
        while proc.poll() is None and (args.loops == 0 or k < args.loops):
            k += 1
            for c in st.values():
                c['total'], c['coll'] = 0.0, 0
            gaps.clear()
            done = spin_until(lambda: st['ego_racecar']['total'] >= run_len, (sync + 25.0 if k == 1 else 30.0) + run_len / 2.0)
            ev = events()
            new, seen = ev[seen:], len(ev)
            started = any(e.startswith('SWITCHBACK') for e in new)
            if gaps and done:
                a = np.array(gaps)
                g0 = float(a[min(int(np.searchsorted(a[:, 0], d_dec)), len(a) - 1), 1])
                g1 = float(a[-1, 1])
                past += g1 < -0.29
                print(f'pass {k:2d}: switchback {"started" if started else "not started"}, gap {g0:+.1f} m before the corner '
                      f'-> {g1:+.1f} m after it ({"the ego is in front" if g1 < -0.29 else "not past"}), '
                      f'contacts {st["ego_racecar"]["coll"]}/{st["opp_racecar"]["coll"]}   [{past} of {k} in front]', flush=True)
            else:
                print(f'pass {k:2d}: not finished (see {log_path})', flush=True)
            # back to the start. First "stand still" (the cars would drive off between the two
            # resets otherwise), then the car ahead, then the ego.
            restart_pub.publish(Float64(data=time.time() + args.pause + 1.0))
            spin_until(lambda: False, 0.3)
            pose.header.stamp = opp_pose.header.stamp = node.get_clock().now().to_msg()
            opp_pub.publish(opp_pose)
            spin_until(lambda: False, 0.3)
            reset_pub.publish(pose)
            spin_until(lambda: False, 0.4)
            for c in st.values():
                c['s'] = None
    except KeyboardInterrupt:
        pass
    finally:
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
        log.close()


if __name__ == '__main__':
    main()
