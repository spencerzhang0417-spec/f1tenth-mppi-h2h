"""Racing lines as sideways offsets from the raceline, and their speed profiles.

An alternative line (late-apex "switchback" line, a defender's inside line) is stored as one
offset d [m, + = left] per raceline waypoint, so it stays index-aligned with the raceline: point i
of every line belongs to the same station. Switching lines is then just using another array, and
blending two lines is a weighted sum.

Shared by the MPPI node (ego), the opponent node and scripts/make_switchback_lines.py. numpy only.
"""
import json

import numpy as np


def resample_closed(xy, ds=0.1):
    """closed polyline -> points every ~ds along it (same scheme as load_waypoints in mppi_node)"""
    xy = np.asarray(xy, dtype=float)
    keep = np.r_[True, np.hypot(*np.diff(xy, axis=0).T) > 1e-6]
    xy = xy[keep]
    if np.hypot(*(xy[0] - xy[-1])) < 1e-3:
        xy = xy[:-1]
    xy_c = np.vstack([xy, xy[:1]])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy_c, axis=0).T))]
    n = int(s[-1] / ds)
    s_new = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.c_[np.interp(s_new, s, xy_c[:, 0]), np.interp(s_new, s, xy_c[:, 1])]


def heading(xy):
    return np.arctan2(np.roll(xy[:, 1], -1) - np.roll(xy[:, 1], 1),
                      np.roll(xy[:, 0], -1) - np.roll(xy[:, 0], 1))


def curvature(xy, smooth=0.5):
    """signed curvature [1/m, + = left] of a closed line, averaged over +-smooth m"""
    yaw = heading(xy)
    seg = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)
    ds = 0.5 * (seg + np.roll(seg, 1))
    dyaw = np.arctan2(np.sin(np.roll(yaw, -1) - np.roll(yaw, 1)), np.cos(np.roll(yaw, -1) - np.roll(yaw, 1)))
    kappa = dyaw / (2.0 * ds)
    win = max(1, int(smooth / float(np.mean(seg))))
    return np.convolve(np.r_[kappa[-win:], kappa, kappa[:win]],
                       np.ones(2 * win + 1) / (2 * win + 1), mode='same')[win:-win]


def speed_profile(xy, a_lat=3.0, a_lon=3.0, v_min=1.5, v_max=6.0):
    """v = clip(sqrt(a_lat / |kappa|), v_min, v_max), then accel / brake <= a_lon around the loop
    (the curvature speed profile of load_waypoints in mppi_node, for any line)"""
    seg = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)       # i -> i+1
    v = np.clip(np.sqrt(a_lat / np.maximum(np.abs(curvature(xy)), 1e-3)), v_min, v_max)
    n = len(v)
    for _ in range(2):
        for i in range(1, 2 * n):          # forward: acceleration limit
            j, k = i % n, (i - 1) % n
            v[j] = min(v[j], np.sqrt(v[k] ** 2 + 2.0 * a_lon * seg[k]))
        for i in range(2 * n - 2, -1, -1):  # backward: braking limit
            j, k = i % n, (i + 1) % n
            v[j] = min(v[j], np.sqrt(v[k] ** 2 + 2.0 * a_lon * seg[j]))
    return np.clip(v, v_min, v_max)


def offset_line(base_xy, d):
    """the raceline moved sideways by d[i] at every point (left normal of the raceline)"""
    yaw = heading(base_xy)
    return np.c_[base_xy[:, 0] - d * np.sin(yaw), base_xy[:, 1] + d * np.cos(yaw)]


def make_waypoints(base_xy, d, **limits):
    """(N, 4) [x, y, yaw, v] of the line `d` away from the raceline"""
    xy = offset_line(base_xy, np.asarray(d, dtype=float))
    return np.c_[xy, heading(xy), speed_profile(xy, **limits)]


def offsets_from_knots(n, ds, knots):
    """smooth d(s) through knots [(s, d), ...] (cubic Hermite with zero slope at the knots, so it
    never overshoots a knot value); zero before the first and after the last knot"""
    s = np.arange(n) * ds
    ks = np.array([k[0] for k in knots], dtype=float)
    kd = np.array([k[1] for k in knots], dtype=float)
    d = np.zeros(n)
    for a in range(len(ks) - 1):
        m = (s >= ks[a]) & (s <= ks[a + 1])
        t = (s[m] - ks[a]) / (ks[a + 1] - ks[a])
        d[m] = kd[a] + (kd[a + 1] - kd[a]) * (t * t * (3.0 - 2.0 * t))
    return d


def lap_times(wps):
    """time [s] at which each waypoint is reached when the speed profile is driven exactly"""
    seg = np.hypot(*(np.roll(wps[:, :2], -1, axis=0) - wps[:, :2]).T)
    dt = seg / np.maximum(0.5 * (wps[:, 3] + np.roll(wps[:, 3], -1)), 1e-3)
    return np.r_[0.0, np.cumsum(dt)[:-1]]


def load_line_set(path):
    """the json written by scripts/make_switchback_lines.py:
    {'ds', 'n', 'corners': [{name, inside, s_decide: [a, b], s_commit, s_end}],
     'attack': [d per waypoint], 'defend': [d per waypoint]}"""
    with open(path) as f:
        data = json.load(f)
    data['attack'] = np.asarray(data['attack'], dtype=float)
    data['defend'] = np.asarray(data['defend'], dtype=float)
    return data


def fit_offsets(d, n):
    """offsets stored for another number of waypoints -> n waypoints (same lap, by fraction)"""
    d = np.asarray(d, dtype=float)
    if len(d) == n:
        return d
    return np.interp(np.arange(n) / n, np.arange(len(d) + 1) / len(d), np.r_[d, d[:1]])
