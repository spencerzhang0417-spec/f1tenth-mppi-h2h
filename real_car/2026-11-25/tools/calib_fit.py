#!/usr/bin/env python3
"""Camera <-> LiDAR rotation from a box seen at two distances (2026-11-25). numpy only.

    python3 calib_fit.py            # prints the fit and the corrected rotation_lidar_to_camera_optical

Input (below): where the box is in the image (pixel edges read from calib/d1.png, d2.png) and which
LiDAR beams hit it (calib/d1.npz, d2.npz, see calib_measure.py). Model, small angles:

    row    = cy + fy * (h_cam - z) / D + pitch_px        box bottom z = 0, box top z = H
    column = cx - fx * (y - camera_y) / D + yaw_px

A wrong height or lateral offset of the camera gives an error that shrinks with distance D; a
rotation gives a constant pixel offset. Two distances separate the two. Result: the offsets are
constant, so the translation measured with a tape (0.071, 0.038, -0.0345) stays and the rotation
changes. Assumes the LiDAR scan plane is parallel to the floor.
"""
import math
import pathlib

import numpy as np

FX, FY, CX, CY = 691.5227661132812, 691.6843872070312, 482.0599365234375, 275.0909729003906
C = np.array([0.071, 0.038, -0.0345])                       # camera lens in the LiDAR frame [m], tape measure
R0 = np.array([0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0]).reshape(3, 3)   # ideal axis swap
# image box edges [px] (left, right, top, bottom), camera depth of the box [m], first / last LiDAR beam on the box
M = {'d1': dict(L=485, R=562, T=241.5, B=379.5, D=1.064, beams=(396, 416)),
     'd2': dict(L=481, R=513, T=281.5, B=337.0, D=2.615, beams=(406, 413))}
HERE = pathlib.Path(__file__).resolve().parent.parent / 'calib'


def main():
    iD = np.array([1.0 / M[k]['D'] for k in ('d1', 'd2')])
    A = np.array([[FY * iD[0], 0, 1], [FY * iD[1], 0, 1], [FY * iD[0], -FY * iD[0], 1], [FY * iD[1], -FY * iD[1], 1]])
    b = np.array([M['d1']['B'], M['d2']['B'], M['d1']['T'], M['d2']['T']]) - CY
    (h_cam, H, pitch_px), *_ = np.linalg.lstsq(A, b, rcond=None)
    print(f"vertical:   camera {h_cam * 100:.1f} cm above the floor, box {H * 100:.1f} cm tall, pitch {pitch_px:.1f} px = {math.degrees(pitch_px / FY):.2f} deg "
          f"(residual {np.sqrt(np.mean((A @ [h_cam, H, pitch_px] - b) ** 2)):.1f} px); LiDAR plane {100 * (h_cam - C[2]):.1f} cm above the floor")
    off = {}
    for k, m in M.items():
        z = np.load(HERE / f'{k}.npz'); ang = float(z['angle_min']) + float(z['angle_increment']) * np.arange(z['ranges'].size)
        bearing = 0.5 * (ang[m['beams'][0]] + ang[m['beams'][1]]); y = (m['D'] + C[0]) * math.tan(bearing)
        off[k] = (CX - FX * (y - C[1]) / m['D']) - 0.5 * (m['L'] + m['R'])
        print(f"horizontal: {k} box at {math.degrees(bearing):+.2f} deg in the scan -> ideal projection is {off[k]:+.1f} px from the box centre in the image")
    dy, yaw_px = np.linalg.solve(np.array([[FX * iD[0], 1], [FX * iD[1], 1]]), -np.array([off['d1'], off['d2']]))
    print(f"            yaw + lateral offset: yaw {math.degrees(yaw_px / FX):.2f} deg, camera_y error {dy * 100:+.1f} cm (noise level) -> yaw only: {math.degrees(-np.mean(list(off.values())) / FX):.2f} deg")
    du, dv = -np.mean(list(off.values())), pitch_px
    beta, alpha = du / FX, -dv / FY
    Ry = np.array([[math.cos(beta), 0, math.sin(beta)], [0, 1, 0], [-math.sin(beta), 0, math.cos(beta)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(alpha), -math.sin(alpha)], [0, math.sin(alpha), math.cos(alpha)]])
    R = Rx @ Ry @ R0
    print(f"correction: scan drawn {du:.1f} px further right and {dv:.1f} px lower than with the ideal matrix\nrotation_lidar_to_camera_optical =")
    print(np.array2string(R, precision=6, suppress_small=True, separator=', '))
    print("needed above the LiDAR plane for a match with the IDEAL matrix (33 px too high): " + ', '.join(f"{d} m: {100 * dv * d / FY:.0f} cm" for d in (0.65, 1.1, 2.0, 3.0)))


if __name__ == '__main__':
    main()
