#!/usr/bin/env python3
"""Live check on the car of what the detector and the LiDAR see (camera must be free; publishes nothing).

    env PATH="/usr/bin:$PATH" PYTHONPATH="$HOME/numpy1_fix:${PYTHONPATH:-}" ~/vision-env/bin/python diag_boxes_and_scan.py <out_dir>

Prints every detection box of the last frame down to score 0.05 (the node uses 0.5), whether a car-sized
box exists, and the projected LiDAR points (beam, u, v, range) across the image centre; saves a zoomed,
annotated crop <out_dir>/diag_crop.jpg. Check the image for people before sharing it.
"""
import sys, time
import numpy as np, cv2, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from vision_lab.camera_capture import V4L2Camera
from vision_lab.detector import TensorRTDetector
from vision_lab.lidar_projection import project_scan
OUT = sys.argv[1]
K = [691.5227661132812, 691.6843872070312, 482.0599365234375, 275.0909729003906]
C_L = [0.071, 0.038, -0.0345]
R_CL = np.array([0.031940, -0.999490, 0.0, 0.048102, 0.001537, -0.998841, 0.998332, 0.031903, 0.048127]).reshape(3, 3)   # calibrated 2026-11-25
rclpy.init(); node = Node('diag_match2'); scan = [None]
node.create_subscription(LaserScan, '/scan', lambda m: scan.__setitem__(0, m), QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
det = TensorRTDetector('/home/nvidia/f1tenth_ws/install/vision_lab/share/vision_lab/models/rtdetrv2_fp16.engine', 80, 0.05)
cam = V4L2Camera(device='/dev/video4', width=960, height=540, fps=60, pixel_format='YUYV')
allb, frame, s = [], None, None
try:
    for _ in range(20): cam.read()
    t_end = time.time() + 15
    while len(allb) < 30 and time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)
        ok, f = cam.read()
        if not ok or scan[0] is None: continue
        frame, s = f, scan[0]
        allb.append(sorted(det.detect(f), key=lambda d: -d['score'])[:8])
finally:
    cam.close(); det.close(); node.destroy_node(); rclpy.shutdown()
print(f"frames {len(allb)}")
print("boxes of the last frame (score, x1, y1, x2, y2, w x h):")
for d in allb[-1]:
    x1, y1, x2, y2 = d['box']; print(f"   {d['score']:.2f}  {x1:5.0f} {y1:5.0f} {x2:5.0f} {y2:5.0f}   {x2-x1:4.0f} x {y2-y1:4.0f}")
tight = [[d for d in fr if (d['box'][2] - d['box'][0]) < 260 and (d['box'][3] - d['box'][1]) < 200] for fr in allb]
bs = [max((d['score'] for d in fr), default=0.0) for fr in tight]
print(f"best score of any car-sized box (w<260, h<200) per frame: median {np.median(bs):.2f} max {max(bs):.2f}; frames with one >= 0.5: {sum(b >= 0.5 for b in bs)}/{len(bs)}")
bb = max((d for fr in tight for d in fr), key=lambda d: d['score'], default=None)
if bb: print("  best car-sized box:", round(bb['score'], 2), [round(v) for v in bb['box']])
bigs = [max(fr, key=lambda d: d['score']) for fr in allb if fr]
print(f"top-scoring box per frame: score median {np.median([d['score'] for d in bigs]):.2f}, width median {np.median([d['box'][2]-d['box'][0] for d in bigs]):.0f} px")
pts = project_scan(s.ranges, s.angle_min, s.angle_increment, s.range_min, s.range_max, *K, C_L, R_CL, 960, 540, include_beam_index=True)
print("\nLiDAR points projected into the image (every 3rd beam), u 250..720:")
print("   beam   u      v     range")
for p in pts[(pts[:, 0] > 250) & (pts[:, 0] < 720)][::3][::-1]:
    print(f"   {int(p[4]):4d}  {p[0]:5.0f}  {p[1]:5.1f}  {p[3]:5.2f}")
img = frame.copy()
for p in pts: cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 255) if p[3] < 3 else (255, 120, 0), -1)
for d in allb[-1][:5]:
    x1, y1, x2, y2 = map(int, d['box']); cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0) if d['score'] >= 0.5 else (0, 0, 255), 1)
cv2.imwrite(f"{OUT}/diag_crop.jpg", cv2.resize(img[180:400, 300:660], None, fx=2.6, fy=2.6, interpolation=cv2.INTER_CUBIC))
