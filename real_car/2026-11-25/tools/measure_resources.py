#!/usr/bin/env python3
"""Sample CPU / GPU / memory on the Jetson while the car software runs. No ROS needed.

    python3 measure_resources.py --duration 60 --out resources/perception_ekf \
        --match perception=car_observations.py --match ekf=opponent_ekf_node.py

Every second: per matched group the summed CPU (percent of ONE core, so 600 = all six cores of the
Orin Nano) and resident memory; from tegrastats (0.5 s): per-core load, GPU load (GR3D_FREQ), RAM,
board power and temperatures. Writes <out>.csv (samples) and <out>.json (mean / p95 / max).
Groups match a regular expression against the full command line; a process counts for the first
group that matches. Use patterns that only the node matches (lib/mppi/follow_node.py, not follow_node.py:
the ros2 launch / ros2 run wrappers carry the bare name too).
"""
import argparse, csv, json, os, re, subprocess, threading, time

import psutil


def tegrastats(lines, stop):
    try:
        p = subprocess.Popen(['tegrastats', '--interval', '500'], stdout=subprocess.PIPE, text=True)
    except FileNotFoundError:
        return
    while not stop.is_set():
        line = p.stdout.readline()
        if not line:
            break
        lines.append((time.time(), line))
    p.terminate()


def parse_tegra(line):
    out = {}
    m = re.search(r'RAM (\d+)/(\d+)MB', line)
    if m: out['ram_used_mb'], out['ram_total_mb'] = int(m[1]), int(m[2])
    m = re.search(r'SWAP (\d+)/(\d+)MB', line)
    if m: out['swap_used_mb'] = int(m[1])
    m = re.search(r'CPU \[([^\]]+)\]', line)
    if m:
        cores = [c for c in m[1].split(',')]
        load = [int(c.split('%')[0]) for c in cores if '%' in c]
        freq = [int(c.split('@')[1]) for c in cores if '@' in c]
        out['cpu_total_pct_of_one_core'] = sum(load)
        out['cpu_max_core_pct'] = max(load) if load else 0
        if freq: out['cpu_freq_mhz'] = max(freq)
    m = re.search(r'GR3D_FREQ (\d+)%', line)
    if m: out['gpu_pct'] = int(m[1])
    for key, name in (('cpu@', 'temp_cpu_c'), ('gpu@', 'temp_gpu_c'), ('tj@', 'temp_tj_c')):
        m = re.search(re.escape(key) + r'([\d.]+)C', line)
        if m: out[name] = float(m[1])
    m = re.search(r'VDD_IN (\d+)mW', line)
    if m: out['power_in_w'] = int(m[1]) / 1000.0
    m = re.search(r'VDD_CPU_GPU_CV (\d+)mW', line)
    if m: out['power_cpu_gpu_w'] = int(m[1]) / 1000.0
    return out


def stats(v):
    v = sorted(v)
    if not v:
        return None
    return {'mean': round(sum(v) / len(v), 2), 'p95': round(v[min(len(v) - 1, int(0.95 * len(v)))], 2), 'max': round(v[-1], 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=float, default=60.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--match', action='append', default=[], help='name=regex on the command line')
    ap.add_argument('--note', default='')
    a = ap.parse_args()
    groups = [(m.split('=', 1)[0], re.compile(m.split('=', 1)[1])) for m in a.match]
    lines, stop = [], threading.Event()
    th = threading.Thread(target=tegrastats, args=(lines, stop), daemon=True); th.start()
    procs, rows, t_end = {}, [], time.time() + a.duration
    own = {os.getpid()} | {q.pid for q in psutil.Process().parents()}
    psutil.cpu_percent(None)
    me = psutil.Process(); me.cpu_percent(None)
    while time.time() < t_end:
        time.sleep(1.0)
        row = {'t': round(time.time(), 2), 'cpu_all_pct_of_one_core': round(psutil.cpu_percent(None) * psutil.cpu_count(), 1)}
        acc = {g: [0.0, 0.0, 0] for g, _ in groups}
        for p in psutil.process_iter(['pid', 'cmdline']):
            try:
                if p.info['pid'] in own:
                    continue                             # this script: its command line contains every pattern
                cmd = ' '.join(p.info['cmdline'] or [])
                g = next((g for g, rx in groups if rx.search(cmd)), None)
                if g is None:
                    continue
                q = procs.setdefault(p.info['pid'], p)
                acc[g][0] += q.cpu_percent(None); acc[g][1] += q.memory_info().rss / 1e6; acc[g][2] += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        for g, (c, r, n) in acc.items():
            row[f'{g}_cpu_pct'], row[f'{g}_rss_mb'], row[f'{g}_n'] = round(c, 1), round(r, 1), n
        row['own_cpu_pct'] = round(me.cpu_percent(None), 1)
        rows.append(row)
    stop.set(); th.join(timeout=2)
    rows = rows[1:]                                      # first per-process sample has no reference interval
    tg = [parse_tegra(l) for _, l in lines]
    with open(a.out + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(a.out + '_tegrastats.log', 'w') as f:
        f.writelines(f'{t:.2f} {l}' for t, l in lines)
    summary = {'note': a.note, 'duration_s': a.duration, 'cores': psutil.cpu_count(), 'samples': len(rows),
               'system': {k: stats([t[k] for t in tg if k in t]) for k in sorted({k for t in tg for k in t})},
               'cpu_all_pct_of_one_core': stats([r['cpu_all_pct_of_one_core'] for r in rows]),
               'this_script_cpu_pct_of_one_core': stats([r['own_cpu_pct'] for r in rows]),
               'groups': {g: {'processes': max(r[f'{g}_n'] for r in rows), 'cpu_pct_of_one_core': stats([r[f'{g}_cpu_pct'] for r in rows]),
                              'rss_mb': stats([r[f'{g}_rss_mb'] for r in rows])} for g, _ in groups}}
    json.dump(summary, open(a.out + '.json', 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
