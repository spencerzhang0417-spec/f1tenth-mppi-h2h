"""EKF for one opponent car, in the ego car's frame (robocentric), fusing three sources:

    car observations           LiDAR surface points of camera-detected cars (lab 10
                               /perception/car_observations, ~15-30 Hz, possibly late)
    VESC odometry              ego speed            (~50 Hz)
    gyro (camera or VESC IMU)  ego yaw rate         (50-200 Hz)

Unlike h2h.OpponentTracker (LiDAR only, map frame, needs the static map and a pose), this
filter needs no map and no localization: the camera says which returns are a car.

Noisy observations are handled in three layers:
    gating         a point is used only if it is within `gate` (chi^2) of the prediction; of
                   several candidates the closest one is used. A beam that slipped past the car
                   onto the wall behind it, or a false positive elsewhere, is rejected.
    confirmation   a new track is "tentative" until `confirm_hits` gated updates arrive within
                   `confirm_window` s; otherwise it is dropped. A lone false positive never
                   becomes a confirmed (published) track.
    robust update  inside the gate, an innovation beyond `huber_k` sigma is scaled back to
                   `huber_k` sigma (Huber): a background hit that just makes the gate moves the
                   estimate a bounded amount instead of kicking the speed by meters per second.
    re-init        `reinit_misses` rejected frames in a row: the track is wrong, start over.

State (ego rear axle, x forward, y left):

    px, py   opponent center relative to the ego          [m]
    phi      opponent heading relative to the ego heading [rad]
    vo       opponent speed over ground, along phi        [m/s]
    wo       opponent yaw rate                            [rad/s]
    ve       ego speed                                    [m/s]
    we       ego yaw rate                                 [rad/s]

Motion (constant turn rate and velocity for both cars, seen from the rotating ego frame):

    px'  = vo cos(phi) - ve + we py
    py'  = vo sin(phi)      - we px
    phi' = wo - we
    vo', wo', ve', we' = white noise

The ego states are measured directly (VESC speed, gyro); the opponent is measured as range and
bearing from the LiDAR origin. vo and phi are absolute in the sense that ego motion is removed:
vo is the opponent's own ground speed. Its heading becomes observable once it moves.

Measurements can arrive late (an observation describes its LiDAR beam time, plus camera
inference, ~50-150 ms before it is received). Every event is stored with the state after it for `history` seconds; a late event is
inserted at its time and everything after it is replayed.

numpy only, no ROS: used by opponent_ekf_node.py and scripts/check_opponent_ekf.py.
"""

import math
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

PX, PY, PHI, VO, WO, VE, WE = range(7)
N = 7
OPP = slice(0, 5)


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class EKFParams:
    # process noise: how fast each quantity can change, as a std per sqrt(s)
    opp_pos_std: float = 0.05        # [m/sqrt(s)] opponent position, what the model misses
    opp_heading_std: float = 0.1     # [rad/sqrt(s)]
    # the two main knobs: larger follows braking / turning faster, smaller gives smoother speed
    opp_accel_std: float = 2.0       # [m/s^2] opponent acceleration
    opp_yaw_accel_std: float = 1.5   # [rad/s^2] opponent yaw acceleration
    ego_accel_std: float = 5.0       # [m/s^2]
    ego_yaw_accel_std: float = 15.0  # [rad/s^2] (the steering moves fast; the gyro is fast too)
    # measurement noise
    speed_std: float = 0.15          # [m/s] VESC speed
    yaw_rate_std: float = 0.03       # [rad/s] gyro
    range_std: float = 0.05          # [m] detection range at 0 m
    range_std_per_m: float = 0.01    # [m/m] grows with range
    bearing_std: float = 0.02        # [rad] box center vs car center (~14 px at fx 690)
    # the LiDAR sees the near surface; the car center is this much further along the ray. Real
    # car: the 12 cm rear box, center 0.23 m ahead of it (mppi_node h2h: depth 0.05 + 0.23)
    surface_offset: float = 0.28     # [m]
    # data association / track management
    gate: float = 11.8               # chi^2 (2 dof): 99.7 %
    huber_k: float = 2.0             # [sigma] innovations beyond this are scaled down; 0 = off
    reinit_misses: int = 5           # this many gated-out detections in a row: start over
    lost_timeout: float = 1.0        # [s] no accepted detection: drop the track
    confirm_hits: int = 3            # gated updates (after the first detection) to confirm
    confirm_window: float = 0.5      # [s] ... within this long, or the tentative track is dropped
    # a new track: the opponent is assumed to drive the ego's way at the ego's speed
    init_heading: float = 0.0        # [rad] relative to the ego
    init_heading_std: float = 0.6
    init_speed_from_ego: bool = True
    init_speed: float = 0.0          # [m/s] used if not init_speed_from_ego
    init_speed_std: float = 1.5
    init_yaw_rate_std: float = 1.0
    # filter
    history: float = 1.0             # [s] how late a measurement may be
    max_dt: float = 0.01             # [s] prediction sub-step


class _State:
    __slots__ = ("t", "x", "P", "active", "t_seen", "misses", "confirmed", "hits", "t_init")

    def __init__(self, t, x, P, active=False, t_seen=-math.inf, misses=0, confirmed=False,
                 hits=0, t_init=-math.inf):
        self.t, self.x, self.P = t, x, P
        self.active, self.t_seen, self.misses = active, t_seen, misses
        self.confirmed, self.hits, self.t_init = confirmed, hits, t_init

    def copy(self):
        return _State(self.t, self.x.copy(), self.P.copy(), self.active, self.t_seen,
                      self.misses, self.confirmed, self.hits, self.t_init)


class _Event:
    __slots__ = ("t", "kind", "data", "after")

    def __init__(self, t, kind, data):
        self.t, self.kind, self.data, self.after = t, kind, data, None


@dataclass
class Estimate:
    t: float
    x: np.ndarray      # (7,) see the state above
    P: np.ndarray      # (7, 7)
    active: bool       # an opponent is being tracked (tentative or confirmed)
    t_seen: float      # time of the last accepted detection
    confirmed: bool    # the track passed confirmation: publish only these


class OpponentEKF:
    def __init__(self, params=None):
        self.p = params or EKFParams()
        p = self.p
        self.Qc = np.diag([p.opp_pos_std ** 2, p.opp_pos_std ** 2, p.opp_heading_std ** 2,
                           p.opp_accel_std ** 2, p.opp_yaw_accel_std ** 2,
                           p.ego_accel_std ** 2, p.ego_yaw_accel_std ** 2])
        self.s = None          # current state (after the newest event)
        self.base = None       # state before the oldest stored event
        self.events = []       # sorted by t
        self.times = []

    # ---------------------------------------------------------------- inputs
    def add_speed(self, t, v):
        """ego speed [m/s] measured at t"""
        return self._add(_Event(float(t), "v", float(v))) is not None

    def add_yaw_rate(self, t, w):
        """ego yaw rate [rad/s], bias removed, measured at t"""
        return self._add(_Event(float(t), "w", float(w))) is not None

    def add_detections(self, t, points, sensor_xy=(0.0, 0.0)):
        """Candidate opponent surface points [(x, y), ...] in the ego frame, best first, seen
        from sensor_xy (the LiDAR origin in the ego frame) at time t.
        Returns 'init', 'update', 'reinit', 'reject', 'none' or 'stale' (older than history)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        sensor = np.asarray(sensor_xy, dtype=np.float64).reshape(2)
        e = self._add(_Event(float(t), "det", (pts, sensor, "")))
        return "stale" if e is None else e.data[2]

    def estimate(self, t=None):
        """Estimate at time t (default: the newest event), predicted from the newest stored state
        at or before t. None before the first measurement."""
        if self.s is None:
            return None
        if t is None or t >= self.s.t:
            s = self.s.copy()
        else:
            i = bisect_right(self.times, t)
            s = (self.events[i - 1].after if i > 0 else self.base).copy()
        if t is not None and t > s.t:
            self._check_lost(s, t)
            self._predict(s, t)
        return Estimate(s.t, s.x, s.P, s.active, s.t_seen, s.active and s.confirmed)

    # ---------------------------------------------------------------- event store
    def _add(self, e):
        if self.s is None:
            x = np.zeros(N)
            P = np.eye(N)
            P[VE, VE], P[WE, WE] = 1.0, 0.25
            # start one history back, so that late measurements from before the first one fit
            self.base = _State(e.t - self.p.history, x, P)
            self.s = self.base.copy()
        if e.t < self.base.t:
            return None
        i = bisect_right(self.times, e.t)
        if i < len(self.events):  # late: restore the state before it and replay
            self.s = (self.events[i - 1].after if i > 0 else self.base).copy()
        self.events.insert(i, e)
        self.times.insert(i, e.t)
        for ev in self.events[i:]:
            self._apply(self.s, ev)
            ev.after = self.s.copy()
        # forget events older than the history
        t_new = self.times[-1]
        k = bisect_right(self.times, t_new - self.p.history)
        if k > 0:
            self.base = self.events[k - 1].after
            del self.events[:k]
            del self.times[:k]
        return e

    def _apply(self, s, e):
        self._check_lost(s, e.t)
        self._predict(s, e.t)
        if e.kind == "v":
            self._update(s, np.array([e.data - s.x[VE]]), _unit(VE), self.p.speed_std ** 2)
        elif e.kind == "w":
            self._update(s, np.array([e.data - s.x[WE]]), _unit(WE), self.p.yaw_rate_std ** 2)
        else:
            pts, sensor = e.data[0], e.data[1]
            e.data = (pts, sensor, self._detect(s, e.t, pts, sensor))

    # ---------------------------------------------------------------- filter steps
    def _check_lost(self, s, t):
        if not s.active:
            return
        timed_out = t - s.t_seen > self.p.lost_timeout
        unconfirmed = not s.confirmed and t - s.t_init > self.p.confirm_window
        if timed_out or unconfirmed:
            s.active = s.confirmed = False
            s.P[OPP, :] = 0.0
            s.P[:, OPP] = 0.0
            s.P[OPP, OPP] = np.eye(5)

    def _predict(self, s, t):
        dt_total = t - s.t
        if dt_total <= 0.0:
            return
        n = max(1, math.ceil(dt_total / self.p.max_dt - 1e-9))
        dt = dt_total / n
        for _ in range(n):
            self._step(s, dt)
        s.t = t

    def _step(self, s, dt):
        x, P = s.x, s.P
        if not s.active:  # only the ego part is meaningful
            P[VE, VE] += self.Qc[VE, VE] * dt
            P[WE, WE] += self.Qc[WE, WE] * dt
            return
        px, py, phi, vo, wo, ve, we = x
        c, sn = math.cos(phi), math.sin(phi)
        F = np.eye(N)
        F[PX, PY] = we * dt
        F[PX, PHI] = -vo * sn * dt
        F[PX, VO] = c * dt
        F[PX, VE] = -dt
        F[PX, WE] = py * dt
        F[PY, PX] = -we * dt
        F[PY, PHI] = vo * c * dt
        F[PY, VO] = sn * dt
        F[PY, WE] = -px * dt
        F[PHI, WO] = dt
        F[PHI, WE] = -dt
        # midpoint integration of the state (heading changes during the step)
        phm = phi + 0.5 * dt * (wo - we)
        x[PX] = px + dt * (vo * math.cos(phm) - ve + we * py)
        x[PY] = py + dt * (vo * math.sin(phm) - we * px)
        x[PHI] = wrap(phi + dt * (wo - we))
        s.P = F @ P @ F.T + self.Qc * dt

    def _update(self, s, y, H, R):
        """Joseph-form update with innovation y, (m, 7) H, (m, m) R"""
        H = np.atleast_2d(H)
        R = np.atleast_2d(R)
        PHt = s.P @ H.T
        S = H @ PHt + R
        K = PHt @ np.linalg.inv(S)
        s.x = s.x + K @ y
        s.x[PHI] = wrap(s.x[PHI])
        A = np.eye(N) - K @ H
        s.P = A @ s.P @ A.T + K @ R @ K.T
        s.P = 0.5 * (s.P + s.P.T)
        # keep vo >= 0 when it is clearly negative: the car drives the other way
        if s.active and s.x[VO] < -max(0.3, 2.0 * math.sqrt(s.P[VO, VO])):
            s.x[VO] = -s.x[VO]
            s.x[PHI] = wrap(s.x[PHI] + math.pi)
            J = np.eye(N)
            J[VO, VO] = -1.0
            s.P = J @ s.P @ J.T

    def _meas(self, pt, sensor):
        """surface point -> (z = [range to the car center, bearing], R)"""
        d = pt - sensor
        r = math.hypot(d[0], d[1])
        z = np.array([r + self.p.surface_offset, math.atan2(d[1], d[0])])
        sr = self.p.range_std + self.p.range_std_per_m * r
        return z, np.diag([sr ** 2, self.p.bearing_std ** 2])

    def _detect(self, s, t, pts, sensor):
        if len(pts) == 0:
            return "none"
        if not s.active:
            self._init_track(s, t, pts[0], sensor)
            return "init"
        dx, dy = s.x[PX] - sensor[0], s.x[PY] - sensor[1]
        r2 = max(dx * dx + dy * dy, 1e-6)
        r = math.sqrt(r2)
        zhat = np.array([r, math.atan2(dy, dx)])
        H = np.zeros((2, N))
        H[0, PX], H[0, PY] = dx / r, dy / r
        H[1, PX], H[1, PY] = -dy / r2, dx / r2
        best = None
        for pt in pts:
            z, R = self._meas(pt, sensor)
            y = z - zhat
            y[1] = wrap(y[1])
            S = H @ s.P @ H.T + R
            d2 = float(y @ np.linalg.solve(S, y))
            if best is None or d2 < best[0]:
                best = (d2, y, R)
        if best[0] <= self.p.gate:
            d2, y, R = best
            m = math.sqrt(d2)
            if self.p.huber_k > 0.0 and m > self.p.huber_k:
                y = y * (self.p.huber_k / m)
            self._update(s, y, H, R)
            s.t_seen, s.misses = t, 0
            s.hits += 1
            if s.hits >= self.p.confirm_hits:
                s.confirmed = True
            return "update"
        s.misses += 1
        if s.misses >= self.p.reinit_misses:
            self._init_track(s, t, pts[0], sensor)
            return "reinit"
        return "reject"

    def _init_track(self, s, t, pt, sensor):
        p = self.p
        (rng, brg), R = self._meas(pt, sensor)
        c, sn = math.cos(brg), math.sin(brg)
        J = np.array([[c, -rng * sn], [sn, rng * c]])
        s.x[PX], s.x[PY] = sensor[0] + rng * c, sensor[1] + rng * sn
        s.x[PHI] = wrap(p.init_heading)
        s.x[VO] = max(s.x[VE], 0.0) if p.init_speed_from_ego else p.init_speed
        s.x[WO] = 0.0
        s.P[OPP, :] = 0.0
        s.P[:, OPP] = 0.0
        s.P[0:2, 0:2] = J @ R @ J.T
        s.P[PHI, PHI] = p.init_heading_std ** 2
        s.P[VO, VO] = p.init_speed_std ** 2
        s.P[WO, WO] = p.init_yaw_rate_std ** 2
        s.active, s.t_seen, s.misses = True, t, 0
        s.confirmed, s.hits, s.t_init = self.p.confirm_hits <= 0, 0, t


def _unit(i):
    H = np.zeros((1, N))
    H[0, i] = 1.0
    return H
