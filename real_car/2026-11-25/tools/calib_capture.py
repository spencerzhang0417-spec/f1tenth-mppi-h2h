"""Capture one camera frame + the matching scan for LiDAR-camera calibration with a box target. Publishes nothing."""
import sys, time
import numpy as np, cv2, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from vision_lab.camera_capture import V4L2Camera
OUT, TAG = sys.argv[1], sys.argv[2]
rclpy.init(); node = Node('calib_capture'); scans = []
node.create_subscription(LaserScan, '/scan', lambda m: scans.append(m), QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
cam = V4L2Camera(device='/dev/video4', width=960, height=540, fps=60, pixel_format='YUYV')
frames = []
try:
    for _ in range(30): cam.read()
    t_end = time.time() + 6
    while (len(frames) < 12 or len(scans) < 20) and time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.03)
        ok, f = cam.read()
        if ok: frames.append(f.astype(np.float32))
finally:
    cam.close(); node.destroy_node(); rclpy.shutdown()
img = np.clip(np.median(np.stack(frames[-12:]), axis=0), 0, 255).astype(np.uint8)       # median of 12 frames: less noise
R = np.array([np.asarray(s.ranges, float) for s in scans[-20:]]); R[~np.isfinite(R) | (R <= 0.05)] = np.nan
rng = np.nanmedian(R, axis=0); s = scans[-1]
cv2.imwrite(f"{OUT}/{TAG}.png", img)
np.savez(f"{OUT}/{TAG}.npz", ranges=rng, angle_min=s.angle_min, angle_increment=s.angle_increment, spread=np.nanstd(R, axis=0))
print(f"saved {TAG}: {len(frames)} frames, {len(scans)} scans")
