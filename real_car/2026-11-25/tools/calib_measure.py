import sys, math
import numpy as np, cv2
OUT, TAG = sys.argv[1], sys.argv[2]
FX, FY, CX, CY = 691.5227661132812, 691.6843872070312, 482.0599365234375, 275.0909729003906
C = np.array([0.071, 0.038, -0.0345])
img = cv2.imread(f"{OUT}/{TAG}.png"); z = np.load(f"{OUT}/{TAG}.npz")
r = z['ranges']; ang = z['angle_min'] + z['angle_increment'] * np.arange(r.size)
def proj(i):
    p = np.array([r[i] * math.cos(ang[i]), r[i] * math.sin(ang[i]), 0.0]) - C
    return FX * (-p[1]) / p[0] + CX, FY * (-p[2]) / p[0] + CY
# target = nearest cluster within +-25 deg
front = np.where((np.abs(ang) < math.radians(25)) & np.isfinite(r))[0]
i_near = front[np.argmin(r[front])]; r0 = r[i_near]
lo = hi = i_near
while np.isfinite(r[lo - 1]) and abs(r[lo - 1] - r[lo]) < 0.08: lo -= 1
while np.isfinite(r[hi + 1]) and abs(r[hi + 1] - r[hi]) < 0.08: hi += 1
xs, ys = r[lo:hi + 1] * np.cos(ang[lo:hi + 1]), r[lo:hi + 1] * np.sin(ang[lo:hi + 1])
width = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
print(f"[{TAG}] LiDAR target cluster: beams {lo}..{hi} ({hi - lo + 1} beams), bearing {math.degrees(ang[lo]):+.2f} .. {math.degrees(ang[hi]):+.2f} deg, range {np.nanmin(r[lo:hi+1]):.3f}..{np.nanmax(r[lo:hi+1]):.3f} m (x {xs.mean():.3f}), width {width:.3f} m")
print(f"   neighbours: beam {lo-1} range {r[lo-1]:.2f}, beam {hi+1} range {r[hi+1]:.2f}")
uR, v_lo = proj(lo); uL, v_hi = proj(hi)       # larger beam index = further left = smaller u
half = FX * z['angle_increment'] / 2
print(f"   projected with the current extrinsics: u {uL:.1f} .. {uR:.1f} (centre {(uL+uR)/2:.1f}, width {uR-uL:.1f} px, +-{half:.1f} px beam quantisation), row v {np.mean([v_lo, v_hi]):.1f}")
# image: cardboard (brown) mask near the projected cluster
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = ((hsv[..., 0] >= 6) & (hsv[..., 0] <= 30) & (hsv[..., 1] >= 45) & (hsv[..., 2] >= 60)).astype(np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)); mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
best = None
for k in range(1, n):
    x, y, w, h, a = st[k]
    if a < 400: continue
    d = abs(x + w / 2 - (uL + uR) / 2)
    if d < 200 and (best is None or a > best[4]): best = (x, y, w, h, a)
if best:
    x, y, w, h, a = best
    print(f"   image: brown blob  u {x} .. {x + w}  (centre {x + w/2:.1f}, width {w} px)   rows {y} .. {y + h}  (height {h} px), area {a}")
    print(f"   => horizontal offset (LiDAR projection - image): left {uL - x:+.1f}, right {uR - (x + w):+.1f}, centre {(uL+uR)/2 - (x + w/2):+.1f} px;  width ratio LiDAR/image {((uR-uL))/w:.2f}")
else: print("   image: no brown blob found near the projection")
# annotated crop with a labelled grid for a visual check
vis = img.copy()
for i in range(lo - 25, hi + 26):
    if np.isfinite(r[i]):
        u, v = proj(i)
        if 0 <= u < 960: cv2.circle(vis, (int(u), int(v)), 2, (0, 255, 255) if lo <= i <= hi else (255, 120, 0), -1)
cu = int((uL + uR) / 2); x0, x1 = max(cu - 200, 0), min(cu + 200, 960); y0, y1 = 120, 460
crop = cv2.resize(vis[y0:y1, x0:x1], None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
for u in range((x0 // 20 + 1) * 20, x1, 20):
    X = int((u - x0) * 2.4); cv2.line(crop, (X, 0), (X, crop.shape[0]), (0, 0, 255) if u % 100 == 0 else (90, 90, 90), 1)
    if u % 40 == 0: cv2.putText(crop, str(u), (X + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
for v in range((y0 // 20 + 1) * 20, y1, 20):
    Y = int((v - y0) * 2.4); cv2.line(crop, (0, Y), (crop.shape[1], Y), (0, 0, 255) if v % 100 == 0 else (90, 90, 90), 1)
    cv2.putText(crop, str(v), (2, Y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
cv2.imwrite(f"{OUT}/{TAG}_grid.jpg", crop)
