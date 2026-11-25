#!/usr/bin/env python3
"""Time each stage of the lab 10 perception node on the car (camera must be free: stop car_observations first).

    env PATH="/usr/bin:$PATH" PYTHONPATH="$HOME/numpy1_fix:${PYTHONPATH:-}" \
        ~/vision-env/bin/python bench_perception.py --frames 300 --out resources/perception_stages.json

Stages as in vision_lab/scripts/car_observations.py: camera.read() (V4L2, blocks for the next frame),
detector.detect() (RT-DETRv2 TensorRT FP16: preprocess + inference + postprocess), project_scan() and
estimate_position() (numpy). Publishes nothing.
"""
import argparse, json, time

import numpy as np
from vision_lab.camera_capture import V4L2Camera
from vision_lab.detector import TensorRTDetector
from vision_lab.lidar_projection import estimate_position, project_scan

K = [691.5227661132812, 691.6843872070312, 482.0599365234375, 275.0909729003906]
C_L = [0.071, 0.038, -0.0345]
R_CL = np.array([0.031940, -0.999490, 0.0, 0.048102, 0.001537, -0.998841, 0.998332, 0.031903, 0.048127]).reshape(3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=300); ap.add_argument('--out', required=True)
    ap.add_argument('--engine', default='/home/nvidia/f1tenth_ws/install/vision_lab/share/vision_lab/models/rtdetrv2_fp16.engine')
    ap.add_argument('--fps', type=int, default=60)
    a = ap.parse_args()
    det = TensorRTDetector(a.engine, 80, 0.5)
    cam = V4L2Camera(device='/dev/video4', width=960, height=540, fps=a.fps, pixel_format='YUYV')
    ranges = (3.0 + 0.5 * np.sin(np.arange(811) / 40.0)).tolist()          # a plausible 811-beam scan
    t = {'camera_read': [], 'detect': [], 'project_scan': [], 'estimate_position': [], 'loop_without_waiting_for_scan': []}
    try:
        for _ in range(30):
            cam.read(); det.detect(cam.read()[1])                          # warm-up
        for _ in range(a.frames):
            t0 = time.perf_counter(); ok, frame = cam.read()
            t1 = time.perf_counter(); dets = det.detect(frame)
            t2 = time.perf_counter(); pts = project_scan(ranges, -2.356, 0.005817, 0.05, 25.0, *K, C_L, R_CL, 960, 540, include_beam_index=True)
            t3 = time.perf_counter()
            for d in dets: estimate_position(d['box'], pts, K, 6.0)
            estimate_position([400, 200, 560, 340], pts, K, 6.0)
            t4 = time.perf_counter()
            for k, v in zip(t, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)): t[k].append(v)
    finally:
        cam.close(); det.close()
    s = lambda v: {'mean_ms': round(1e3 * float(np.mean(v)), 2), 'p50_ms': round(1e3 * float(np.percentile(v, 50)), 2), 'p95_ms': round(1e3 * float(np.percentile(v, 95)), 2), 'max_ms': round(1e3 * float(np.max(v)), 2)}
    out = {'frames': a.frames, 'camera': f'/dev/video4 960x540 YUYV @ {a.fps} fps', 'engine': a.engine.split('/')[-1],
           'stages': {k: s(v) for k, v in t.items()}, 'free_running_rate_hz': round(1.0 / float(np.mean(t['loop_without_waiting_for_scan'])), 1)}
    json.dump(out, open(a.out, 'w'), indent=2); print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
