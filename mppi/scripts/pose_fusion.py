"""High-rate pose: the particle filter's pose brought up to now with wheel speed and gyro.

The particle filter (laser against the map) is accurate but late: the laser turns at 15 Hz and
its pose describes the car ~0.125 s ago. The VESC speed and the IMU gyro arrive at 50 Hz with
almost no delay, but only say how the car moved, not where it is. PoseFuser joins them:

    add_motion(t, v, w)       speed [m/s] and yaw rate [rad/s], every time one arrives
    add_pose(t, x, y, yaw)    particle-filter pose RECEIVED at t: it is taken as the pose at
                              t - lag, then moved forward through the motion recorded since
    pose(t)                   the pose at time t (normally: now)

Between two particle-filter poses the estimate is pure dead reckoning for ~0.2 s at most, so it
cannot drift far, and every new particle-filter pose pulls it back.

The tracked point need not be the rear axle: with lever = its distance ahead of the rear axle
(0.27 m for the laser pose the particle filter publishes on the car) the point also moves
sideways when the car turns, v_point = v * heading + w * lever * left.

Plain numpy / math, no ROS: used by pose_fusion_node.py on the car and by the offline bag check
(scripts/check_pose_fusion.py).
"""
from collections import deque

import numpy as np


def _advance(x, y, yaw, v, w, dt, lever):
    """move a point `lever` ahead of the rear axle for dt with speed v and yaw rate w (midpoint)"""
    ym = yaw + 0.5 * w * dt
    c, s = np.cos(ym), np.sin(ym)
    return (x + dt * (v * c - w * lever * s),
            y + dt * (v * s + w * lever * c),
            yaw + w * dt)


class PoseFuser:
    def __init__(self, lag=0.125, lever=0.0, history=2.0, blend=1.0):
        """
        lag     [s] age of a particle-filter pose when it is received
        lever   [m] the pose point ahead of the rear axle
        history [s] of motion samples kept
        blend   1.0 = jump to each new particle-filter pose; < 1 takes that fraction of the
                correction per pose (smoother, slower to follow the filter)
        """
        self.lag, self.lever, self.blend = float(lag), float(lever), float(blend)
        self.history = float(history)
        self.motion = deque()      # (t, v, w), each valid from its t until the next sample
        self.anchor = None         # (t, x, y, yaw): pose at time t
        self.last_correction = (0.0, 0.0, 0.0)  # what the newest particle-filter pose changed

    def add_motion(self, t, v, w):
        self.motion.append((float(t), float(v), float(w)))
        while self.motion and self.motion[0][0] < t - self.history:
            self.motion.popleft()

    def _propagate(self, pose, t0, t1):
        """pose at t0 -> pose at t1 (t1 >= t0) through the recorded motion (zero-order hold)"""
        x, y, yaw = pose
        if t1 <= t0 or not self.motion:
            return x, y, yaw
        m = self.motion
        t = t0
        # the sample in force at t0: the last one at or before it (else the first one)
        k = 0
        for i in range(len(m)):
            if m[i][0] <= t0:
                k = i
            else:
                break
        while t < t1:
            t_next = m[k + 1][0] if k + 1 < len(m) else np.inf
            t_end = min(t1, max(t_next, t))
            x, y, yaw = _advance(x, y, yaw, m[k][1], m[k][2], t_end - t, self.lever)
            t = t_end
            if t_next <= t and k + 1 < len(m):
                k += 1
        return x, y, yaw

    def add_pose(self, t_recv, x, y, yaw):
        t_meas = t_recv - self.lag
        if self.anchor is not None:
            # where the estimate was at the time this pose describes
            ex, ey, eyaw = self._propagate(self.anchor[1:], self.anchor[0], t_meas) \
                if t_meas >= self.anchor[0] else self.anchor[1:]
            dyaw = np.arctan2(np.sin(yaw - eyaw), np.cos(yaw - eyaw))
            self.last_correction = (x - ex, y - ey, dyaw)
            if self.blend < 1.0 and t_meas >= self.anchor[0]:
                x, y, yaw = ex + self.blend * (x - ex), ey + self.blend * (y - ey), eyaw + self.blend * dyaw
        self.anchor = (t_meas, x, y, yaw)

    def pose(self, t):
        """(x, y, yaw) at time t, or None before the first particle-filter pose"""
        if self.anchor is None:
            return None
        x, y, yaw = self._propagate(self.anchor[1:], self.anchor[0], t)
        return x, y, np.arctan2(np.sin(yaw), np.cos(yaw))
