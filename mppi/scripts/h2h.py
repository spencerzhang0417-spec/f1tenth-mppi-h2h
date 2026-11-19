"""Head-to-head racing on top of the MPPI node: see other cars, follow them, pass them.

Three pieces, all plain numpy (the MPPI cost for moving cars lives in mppi_node.py):

  TrackFrame       the raceline as a Frenet frame (s along it, d to its left) plus the free width
                   on each side, measured from the static map
  OpponentTracker  LiDAR-only: scan returns the static map cannot explain -> clusters of car size
                   -> one Kalman filter (position + velocity) per car
  Tactician        FOLLOW (keep a safe gap) or PASS (left / right) for the nearest car ahead

Everything the simulator knows about the other cars (ground-truth odometry) is deliberately NOT
used: on the real car the LiDAR is all there is.

Conventions: positions are car CENTERS in the map frame; d > 0 is left of the raceline;
gap = s_opponent - s_ego (center to center), so gap > 0 means the opponent is ahead.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

CAR_LENGTH = 0.58
CAR_WIDTH = 0.31


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class TrackFrame:
    def __init__(self, waypoints):
        """waypoints: (N, 4) [x, y, yaw, v] of a closed line, uniformly spaced"""
        self.wps = np.asarray(waypoints, dtype=float)
        self.n = len(self.wps)
        self.ds = float(np.mean(np.hypot(np.diff(self.wps[:, 0]), np.diff(self.wps[:, 1]))))
        self.length = self.n * self.ds
        yaw = self.wps[:, 2]
        dyaw = np.arctan2(np.sin(np.roll(yaw, -1) - np.roll(yaw, 1)), np.cos(np.roll(yaw, -1) - np.roll(yaw, 1)))
        kappa = dyaw / (2.0 * self.ds)
        win = max(1, int(0.3 / self.ds))  # ~0.6 m moving average
        self.kappa = np.convolve(np.r_[kappa[-win:], kappa, kappa[:win]],
                                 np.ones(2 * win + 1) / (2 * win + 1), mode='same')[win:-win]
        # free width left / right of the line; "very wide" until the map arrives
        self.w_left = np.full(self.n, 5.0)
        self.w_right = np.full(self.n, 5.0)
        self.wall_dist = None  # (rows, cols) distance to the nearest wall / unknown cell [m]
        self.res = self.ox = self.oy = None

    # ---- static map ----
    def set_map(self, wall, res, ox, oy, w_max=5.0, step=0.02):
        """wall: (rows, cols) bool, True = wall or unknown; cell (r, c) covers x >= ox + c*res"""
        wall = np.asarray(wall, dtype=bool)
        self.res, self.ox, self.oy = float(res), float(ox), float(oy)
        self.wall_dist = (distance_transform_edt(~wall) * self.res).astype(np.float32)
        nx, ny = -np.sin(self.wps[:, 2]), np.cos(self.wps[:, 2])
        d = np.arange(0.0, w_max, step)
        for sign, out in ((1.0, self.w_left), (-1.0, self.w_right)):
            px = self.wps[:, 0:1] + sign * d[None, :] * nx[:, None]
            py = self.wps[:, 1:2] + sign * d[None, :] * ny[:, None]
            blocked = self.clearance(px, py) <= 0.0
            out[:] = d[np.where(blocked.any(axis=1), blocked.argmax(axis=1), len(d) - 1)]

    def clearance(self, x, y):
        """distance to the nearest wall of the static map [m]; 0 off the map, inf without a map"""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if self.wall_dist is None:
            return np.full(x.shape, np.inf)
        c = np.floor((x - self.ox) / self.res).astype(int)
        r = np.floor((y - self.oy) / self.res).astype(int)
        ok = (r >= 0) & (r < self.wall_dist.shape[0]) & (c >= 0) & (c < self.wall_dist.shape[1])
        out = np.zeros(x.shape, dtype=float)
        out[ok] = self.wall_dist[r[ok], c[ok]]
        return out

    # ---- Frenet ----
    def nearest(self, x, y):
        """index of the nearest waypoint (x, y scalars or arrays)"""
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        return np.argmin(np.hypot(self.wps[None, :, 0] - x[:, None], self.wps[None, :, 1] - y[:, None]), axis=1)

    def frenet(self, x, y):
        """(s, d) of one point"""
        i = int(self.nearest(x, y)[0])
        wx, wy, yaw = self.wps[i, 0], self.wps[i, 1], self.wps[i, 2]
        dx, dy = x - wx, y - wy
        s = (i * self.ds + dx * np.cos(yaw) + dy * np.sin(yaw)) % self.length
        return s, -dx * np.sin(yaw) + dy * np.cos(yaw)

    def pose(self, s, d):
        """(x, y, yaw) at station(s) s with sideways offset d"""
        s = np.asarray(s, dtype=float) % self.length
        f = s / self.ds
        i = np.floor(f).astype(int) % self.n
        j = (i + 1) % self.n
        t = f - np.floor(f)
        dyaw = np.arctan2(np.sin(self.wps[j, 2] - self.wps[i, 2]), np.cos(self.wps[j, 2] - self.wps[i, 2]))
        yaw = self.wps[i, 2] + t * dyaw
        x = (1.0 - t) * self.wps[i, 0] + t * self.wps[j, 0] - d * np.sin(yaw)
        y = (1.0 - t) * self.wps[i, 1] + t * self.wps[j, 1] + d * np.cos(yaw)
        return x, y, yaw

    def wrap(self, ds):
        """station difference wrapped to -L/2 .. L/2"""
        return (np.asarray(ds, dtype=float) + 0.5 * self.length) % self.length - 0.5 * self.length

    def window(self, s0, s1):
        """waypoint indices covering stations s0 .. s1 (s1 may wrap past the finish line)"""
        i0 = int(np.floor((s0 % self.length) / self.ds))
        m = max(int(np.ceil(self.wrap(s1 - s0) / self.ds)), 1) if s1 != s0 else 1
        return (i0 + np.arange(m + 1)) % self.n


class Track:
    """one tracked car: constant-velocity Kalman filter on [x, y, vx, vy]"""
    _next_id = 0

    def __init__(self, x, y, t, vx=0.0, vy=0.0):
        self.x = np.array([x, y, vx, vy], dtype=float)
        self.P = np.diag([0.1, 0.1, 9.0, 9.0])
        self.t, self.born, self.last_seen, self.hits = t, t, t, 1
        self.beams = np.zeros(0, dtype=int)
        self.id = Track._next_id
        Track._next_id += 1
        self.s = self.d = self.vs = self.vd = 0.0   # station, offset, speed along / across the raceline
        self.c = (x, y)        # the car's center (the measured point may be its rear box)
        self.moving = False

    def predict(self, t, accel_std):
        dt = max(t - self.t, 0.0)
        self.t = t
        if dt <= 0.0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = accel_std ** 2
        Q = q * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0], [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                          [dt ** 3 / 2, 0, dt ** 2, 0], [0, dt ** 3 / 2, 0, dt ** 2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, meas_std):
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        S = H @ self.P @ H.T + np.eye(2) * meas_std ** 2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
        self.last_seen = self.t
        self.hits += 1


class OpponentTracker:
    def __init__(self, frame, wall_margin=0.25, max_range=6.0, cluster_gap=0.3, min_points=3,
                 max_extent=0.9, depth=0.2, gate=0.8, accel_std=4.0, meas_std=0.08,
                 confirm_hits=5, coast_time=0.6, moving_speed=0.4, center_ahead=0.0):
        self.frame = frame
        self.wall_margin, self.max_range = wall_margin, max_range
        self.cluster_gap, self.min_points, self.max_extent = cluster_gap, min_points, max_extent
        self.depth, self.gate = depth, gate
        self.accel_std, self.meas_std = accel_std, meas_std
        self.confirm_hits, self.coast_time, self.moving_speed = confirm_hits, coast_time, moving_speed
        # what the laser sees of a car is its rear box (12 x 12 cm, on every car by the rules): the
        # car's center is center_ahead m in front of it, along the car's heading (0 = the cluster
        # is the car body itself)
        self.center_ahead = center_ahead
        self.tracks = []

    def detect(self, ranges, angles, lx, ly, yaw):
        """clusters of scan returns that are not walls of the map: [(cx, cy, beam indices)]"""
        ok = np.isfinite(ranges) & (ranges > 0.2) & (ranges < self.max_range)
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            return []
        ca, sa = np.cos(yaw + angles[idx]), np.sin(yaw + angles[idx])
        wx, wy = lx + ranges[idx] * ca, ly + ranges[idx] * sa
        keep = self.frame.clearance(wx, wy) > self.wall_margin
        idx, wx, wy, ca, sa = idx[keep], wx[keep], wy[keep], ca[keep], sa[keep]
        if len(idx) == 0:
            return []
        # split where two consecutive kept returns are far apart
        cut = np.nonzero(np.hypot(np.diff(wx), np.diff(wy)) > self.cluster_gap)[0] + 1
        out = []
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(idx)]):
            if b - a < self.min_points:
                continue
            if np.hypot(wx[b - 1] - wx[a], wy[b - 1] - wy[a]) > self.max_extent:
                continue
            # the returns are the car's visible faces; its center is ~depth further along the beams
            mx, my = wx[a:b].mean(), wy[a:b].mean()
            ux, uy = ca[a:b].mean(), sa[a:b].mean()
            nrm = max(np.hypot(ux, uy), 1e-6)
            out.append((mx + self.depth * ux / nrm, my + self.depth * uy / nrm, idx[a:b]))
        return out

    def update(self, ranges, angles, lx, ly, yaw, t):
        """one scan taken at lidar position (lx, ly) and heading yaw, at time t [s]"""
        dets = self.detect(ranges, angles, lx, ly, yaw)
        for tr in self.tracks:
            tr.predict(t, self.accel_std)
            tr.beams = np.zeros(0, dtype=int)
        # greedy nearest-neighbour association, closest pairs first
        pairs = sorted((np.hypot(tr.x[0] - cx, tr.x[1] - cy), ti, di)
                       for ti, tr in enumerate(self.tracks) for di, (cx, cy, _) in enumerate(dets))
        used_t, used_d = set(), set()
        for dist, ti, di in pairs:
            if dist > self.gate or ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self.tracks[ti].update(dets[di][:2], self.meas_std)
            self.tracks[ti].beams = dets[di][2]
        for di, (cx, cy, beams) in enumerate(dets):
            if di not in used_d:
                tr = Track(cx, cy, t)
                tr.beams = beams
                self.tracks.append(tr)
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen < self.coast_time]
        for tr in self.tracks:
            # tr.x is what was measured (the box); tr.c the car's center, ahead of it along the
            # direction the car drives
            s_box, _ = self.frame.frenet(tr.x[0], tr.x[1])
            # (the raceline's direction at the box, not the velocity estimate: a 12 cm box gives
            # few returns and the velocity's direction is too noisy to lever 0.23 m off it)
            yaw_c = self.frame.wps[int(s_box / self.frame.ds) % self.frame.n, 2]
            tr.c = (tr.x[0] + self.center_ahead * np.cos(yaw_c), tr.x[1] + self.center_ahead * np.sin(yaw_c))
            tr.s, tr.d = self.frame.frenet(tr.c[0], tr.c[1])
            yaw_s = self.frame.wps[int(tr.s / self.frame.ds) % self.frame.n, 2]
            tr.vs = float(tr.x[2] * np.cos(yaw_s) + tr.x[3] * np.sin(yaw_s))
            tr.vd = float(-tr.x[2] * np.sin(yaw_s) + tr.x[3] * np.cos(yaw_s))
            # hysteresis, so a car at about the threshold does not flip between car and obstacle
            tr.moving = tr.vs > (0.6 if tr.moving else 1.0) * self.moving_speed
        return self.confirmed()

    def confirmed(self):
        return [tr for tr in self.tracks if tr.hits >= self.confirm_hits]

    @staticmethod
    def mask_scan(ranges, angles, lx, ly, yaw, tracks, radius=0.5, shift=(0.0, 0.0)):
        """the scan without the returns of the given tracks: beams that end within `radius` of a
        track get the range of the neighbouring beams (the wall behind the car), so the static
        obstacle code does not see moving cars. Geometric, not by beam index: the scan given here
        may be newer than the one the tracks were last updated with."""
        if not tracks:
            return ranges
        ok = np.isfinite(ranges)
        r = np.where(ok, ranges, 0.0)
        wx, wy = lx + r * np.cos(yaw + angles), ly + r * np.sin(yaw + angles)
        mask = np.zeros(len(ranges), dtype=bool)
        for tr in tracks:
            mask |= ok & (np.hypot(wx - tr.x[0] - shift[0], wy - tr.x[1] - shift[1]) < radius)
        good = ~mask & ok
        if not mask.any() or good.sum() < 2:
            return ranges
        out = np.array(ranges, dtype=float)
        k = np.arange(len(out))
        out[mask] = np.interp(k[mask], k[good], out[good])
        return out


class Tactician:
    """FOLLOW or PASS for the nearest moving car ahead.

    PASS is only started on a straight enough stretch with room next to the opponent, and only
    when the ego is meaningfully faster there. Otherwise FOLLOW: an adaptive-cruise speed cap that
    keeps `follow_gap + follow_time * v` m bumper to bumper.
    """
    FREE, FOLLOW, PASS, SWITCHBACK = 'FREE', 'FOLLOW', 'PASS', 'SWITCHBACK'

    def __init__(self, frame, lat_sep=0.7, min_lat_sep=0.55, wall_margin=0.12, pass_kappa=0.4,
                 min_dv=0.4, look=4.0, follow_gap=0.3, follow_time=0.2, follow_gain=1.5,
                 pass_timeout=8.0, cooldown=1.5, overlap_gap=0.8, clear_ahead=0.6, max_pass_time=5.0,
                 move_kappa=0.6, ego_accel=2.0, follow_decel=1.0, react_time=0.35, follow_step=0.3,
                 corners=(), sb_inside=0.2, sb_max_gap=3.0, sb_speed=0.8, sb_give_up=0.15,
                 sb_commit=False, claim=True, claim_dv=0.2, pass_min_time=0.8, brake_look=2.5,
                 pass_look=3.5, pass_path_kappa=0.35):
        self.f = frame
        self.lat_sep, self.min_lat_sep, self.wall_margin = lat_sep, min_lat_sep, wall_margin
        self.pass_kappa, self.min_dv, self.look = pass_kappa, min_dv, look
        self.follow_gap, self.follow_time, self.follow_gain = follow_gap, follow_time, follow_gain
        self.pass_timeout, self.cooldown = pass_timeout, cooldown
        self.overlap_gap, self.clear_ahead, self.max_pass_time = overlap_gap, clear_ahead, max_pass_time
        self.move_kappa, self.ego_accel = move_kappa, ego_accel
        self.follow_decel, self.react_time, self.follow_step = follow_decel, react_time, follow_step
        # switchback: corners with a late-apex line (scripts/make_apex_lines.py); empty = off
        self.corners, self.sb_inside, self.sb_max_gap, self.sb_speed = list(corners), sb_inside, sb_max_gap, sb_speed
        self.sb_give_up = sb_give_up
        self.sb_commit = sb_commit   # True: never give a started switchback up (to test the line itself)
        self.claim_on, self.claim_dv = claim, claim_dv
        self.pass_min_time, self.brake_look = pass_min_time, brake_look
        self.pass_look, self.pass_path_kappa = pass_look, pass_path_kappa
        self.claim = False     # the ego holds the inside against the target right now
        self.corner = None     # the corner of the running switchback
        self.sb_count = 0      # switchbacks started / given up before the turn-in
        self.sb_aborted = 0
        self.state, self.side, self.target = self.FREE, 0, None
        self.t_pass = self.t_block = self.t_start = -1e9
        self.d_pass = 0.0   # sideways position (Frenet d) the ego passes at
        self.v_cap = np.inf
        self.info = ''
        self.changed = False
        self.gap = np.nan   # to the car being followed / passed [m]

    def room(self, tr, side, s0, s1):
        """smallest sideways distance from the opponent's center to the wall on `side`
        (+1 left, -1 right) over stations s0 .. s1 [m]"""
        idx = self.f.window(s0, s1)
        w = self.f.w_left[idx] if side > 0 else self.f.w_right[idx]
        return float(np.min(w - side * tr.d))

    def step(self, s_e, d_e, v_e, tracks, t):
        f = self.f
        # the nearest moving car ahead (or alongside / just passed while a pass is running)
        best, best_gap = None, np.inf
        for tr in tracks:
            if not tr.moving:
                continue
            gap = float(f.wrap(tr.s - s_e))
            lo = -3.0 if (self.state == self.PASS and tr is self.target) else -0.3
            if lo < gap < self.look and gap < best_gap:
                best, best_gap = tr, gap
        if self.state == self.PASS and self.target is not None and self.target in tracks:
            g = float(f.wrap(self.target.s - s_e))
            if -3.0 < g < self.look:
                best, best_gap = self.target, g  # finish the pass that is running
        self.v_cap = np.inf
        self.changed = False
        self.gap = np.nan if best is None else best_gap
        self.claim = False
        if self._switchback(s_e, v_e, best, best_gap, t):
            return
        if best is None:
            self._set(self.FREE, 0, None, t, '')
            return
        if self._hold_inside(s_e, d_e, v_e, best, best_gap, t):
            return
        tr, gap = best, best_gap
        v_free = float(np.mean(f.wps[f.window(s_e, s_e + 6.0), 3]))  # what the ego could drive here
        dv = v_free - tr.vs
        need = self.min_lat_sep + 0.5 * CAR_WIDTH + self.wall_margin  # opponent center -> wall

        if self.state == self.PASS and tr is self.target:
            # running pass: keep the side. Give up while the ego is still behind the opponent's
            # center if the room closes before it can get clear, or if it is not getting past
            # Looks only at the room beside the opponent now and just ahead, and not before the
            # pass is pass_min_time old. Re-predicting the whole side-by-side stretch every step,
            # as the start check does, flips with every wobble of the speed estimate when the ego is
            # only a little faster: passes started and were given up within 20 ms, and each flip
            # moves the reference line by half a meter -- the car swung left and right out of the
            # hairpin behind the other car.
            room_now = self.room(tr, self.side, tr.s - 0.3, tr.s + 2.5)
            young = t - self.t_start < self.pass_min_time
            if gap > 0.0 and not young and (room_now < need - 0.1 or t - self.t_pass > self.pass_timeout):
                self.t_block = t
                self._set(self.FOLLOW, 0, tr, t, f'pass aborted (room {room_now:.2f} m, gap {gap:+.1f} m)')
            else:
                self.d_pass = self._pass_line(tr, self.side, room_now)
                self.info = f'gap {gap:+.1f} m'
        else:
            # Only the side-by-side phase needs a straight with room: from the moment the cars
            # start to overlap (gap = overlap_gap) until the ego is clear ahead (gap = -clear_ahead).
            # Where will the opponent be during that phase, if both keep their speeds?
            s_a, s_b, t_b = self._side_by_side(tr, gap, v_e, v_free)
            idx = f.window(s_a, s_b)
            kap = float(np.max(np.abs(f.kappa[idx])))
            kap_e = float(np.max(np.abs(f.kappa[f.window(s_e, s_e + 1.5)])))
            kap_path = float(np.max(np.abs(f.kappa[f.window(s_e, s_e + self.pass_look)])))
            rooms = {side: self.room(tr, side, s_a, s_b) for side in (1, -1)}
            side = max(rooms, key=rooms.get)
            if dv <= self.min_dv or t_b >= self.max_pass_time:
                why = f'too slow (dv {dv:.1f} m/s)'
            elif kap >= self.pass_kappa:
                why = f'corner (kappa {kap:.2f})'
            elif kap_e >= self.move_kappa:
                # the ego itself is still turning: it has no steering left to change lanes
                # (and a move to the inside of a hairpin is tighter than the car can turn)
                why = f'ego in corner (kappa {kap_e:.2f})'
            elif kap_path >= self.pass_path_kappa:
                # a bend within the next few meters of the ego's own path. Pulling out of line
                # while accelerating out of one corner into the next put the car 0.5 m/s over the
                # speed of that bend and 1.4 m off the raceline: it slid into the wall (twice
                # in two runs, at the right kink after the first hairpin)
                why = f'bend ahead (kappa {kap_path:.2f})'
            elif rooms[side] < need:
                why = f'no room ({rooms[side]:.2f} m)'
            elif t - self.t_block <= self.cooldown:
                why = 'cooldown'
            else:
                why = None
            if why is None:
                self.t_pass = self.t_start = t
                self.d_pass = self._pass_line(tr, side, rooms[side])
                self._set(self.PASS, side, tr, t,
                          f'{"left" if side > 0 else "right"}: gap {gap:+.1f} m, dv {dv:.1f} m/s, '
                          f'room {rooms[side]:.2f} m')
            else:
                self._set(self.FOLLOW, 0, tr, t, why)

        if self.state == self.FOLLOW:
            gap_b = gap - CAR_LENGTH
            want = self.follow_gap + self.follow_time * max(v_e, 0.0)
            # the speed the car ahead is about to have, not the one it has: it will brake for the
            # corner in front of it as the ego's own profile does there, scaled by how fast it is
            # driving relative to that profile now. With the short gap and only its current speed,
            # the ego was still 0.7 m/s faster when the car ahead braked for the hairpin, and
            # touched its tail (bumper gap 5 cm, one contact in 50 s behind a 0.92x car).
            idx_o = f.window(tr.s, tr.s + self.brake_look)
            ratio = float(np.clip(tr.vs / max(f.wps[idx_o[0], 3], 0.5), 0.5, 1.2))
            v_o = min(tr.vs, ratio * float(np.min(f.wps[idx_o, 3])))
            v_lin = v_o + self.follow_gain * (gap_b - want)
            # and never faster than the ego can still brake down to the opponent's speed in the gap
            # it has, after its reaction time (pose age + control step + actuation). The linear law
            # alone starts braking too late when closing fast: with the car's 0.3 s of delay the
            # ego arrived 1 m/s too fast, swerved around the opponent and stopped at the wall
            free = gap_b - self.follow_gap - max(v_e - v_o, 0.0) * self.react_time
            v_brk = np.sqrt(max(v_o, 0.0) ** 2 + 2.0 * self.follow_decel * max(free, 0.0))
            # and never much above the ego's own speed: MPPI accelerates at its limit (+2.5 m/s^2)
            # up to whatever the cap allows, and with 0.3 s of delay it then overshoots a cap that
            # drops as the gap closes (0.7 m/s over it out of a hairpin, then a swerve around the car)
            v_soft = max(v_e, 0.0) + self.follow_step
            self.v_cap = max(min(v_lin, v_brk, v_soft), 0.0)

    def _switchback(self, s_e, v_e, tr, gap, t):
        """the switchback: when the car ahead covers the inside before a corner, take the late-apex
        line through it (wide in, late apex, out on the inside) instead of following it.
        True while it is running: the caller then uses the late-apex line and no speed cap."""
        f = self.f
        if self.state == self.SWITCHBACK:
            c = self.corner
            zone = (c['s_end'] - c['s_split']) % f.length
            if float(f.wrap(s_e - c['s_split'])) > zone:        # back on the raceline: done
                self.corner = None
                self._set(self.FREE, 0, None, t, f'switchback {c["name"]} finished, gap {gap:+.1f} m')
                return False
            before_commit = float(f.wrap(s_e - c['s_commit'])) < 0.0
            opp_before_apex = tr is not None and float(f.wrap(tr.s - c['s_apex'])) < -0.5
            if not self.sb_commit and before_commit and opp_before_apex and gap > 0.3 \
                    and c['inside'] * tr.d < -self.sb_give_up:
                # the car ahead went to the OUTSIDE after all, where the late-apex line goes:
                # nothing to cut under, follow it. Only while the opponent has not reached the apex:
                # past it, running wide is exactly what an early-apex car does and what the
                # late-apex line cuts under. (Two earlier versions ended every switchback within a
                # second: one gave up when the opponent was < 5 cm inside, the other when it ran
                # wide AFTER its apex.)
                self.corner = None
                self.sb_aborted += 1
                self.t_block = t
                self._set(self.FOLLOW, 0, tr, t, f'switchback {c["name"]} given up: opponent at {tr.d:+.2f} m')
                return False
            self.info = f'{c["name"]} gap {gap:+.1f} m'
            return True
        if tr is None or self.state == self.PASS or not self.corners:
            return False
        for c in self.corners:
            a, b = c['s_decide']
            if (s_e - a) % f.length > (b - a) % f.length:
                continue                                        # not in this corner's decision stretch
            inside_now = c['inside'] * (tr.d + 0.4 * tr.vd)     # where it is heading in 0.4 s
            v_line = float(f.wps[int(s_e / f.ds) % f.n, 3])
            if 0.3 < gap < self.sb_max_gap and inside_now >= self.sb_inside \
                    and v_e >= self.sb_speed * v_line and v_e >= tr.vs - 0.3:
                self.corner = c
                self.sb_count += 1
                self._set(self.SWITCHBACK, -c['inside'], tr, t,
                          f'{c["name"]}: opponent {c["inside"] * tr.d:+.2f} m inside, gap {gap:+.1f} m, '
                          f'entry speed {v_e:.1f} of {v_line:.1f} m/s')
                return True
        return False

    def _hold_inside(self, s_e, d_e, v_e, tr, gap, t):
        """right of way: the ego is next to the car (overlapping or ahead), on the inside of the
        corner ahead, and faster (or its nose is already in front). Then the corner is the ego's:
        it keeps its line beside the car and its speed -- no following, no giving the pass up --
        and the other car is the one that has to leave room. Before this rule the ego braked
        every time the car on the outside started to close the door, and gave the place back."""
        if not self.claim_on:
            return False
        f = self.f
        side = 1 if d_e > tr.d else -1                      # the ego is on this side of the car
        k = f.kappa[f.window(s_e, s_e + 8.0)]
        k_big = float(k[int(np.argmax(np.abs(k)))])
        inside = int(np.sign(k_big)) if abs(k_big) > 0.25 else 0   # inside of the corner ahead (0: none)
        beside = gap < self.overlap_gap and abs(d_e - tr.d) > 0.25
        if not (beside and side == inside and (v_e > tr.vs + self.claim_dv or gap < 0.0)):
            return False
        self.claim = True
        if self.state != self.PASS:
            self.t_start = t
        self.t_pass = t
        # stay beside it: as far apart as now (at least half a car more than touching), inside the track
        idx = f.window(tr.s - 0.3, tr.s + 1.5)
        wall = float(np.min(f.w_left[idx] if side > 0 else f.w_right[idx])) - 0.5 * CAR_WIDTH - self.wall_margin
        sep = float(np.clip(abs(d_e - tr.d), self.min_lat_sep - 0.1, self.lat_sep))
        self.d_pass = side * min(side * (tr.d + side * sep), wall)
        self._set(self.PASS, side, tr, t, f'holding the inside ({"left" if side > 0 else "right"}), gap {gap:+.1f} m')
        return True

    def _time_to_gain(self, dist, v_e, v_o, v_free):
        """time [s] for the ego to gain `dist` m on a car driving v_o, accelerating with ego_accel
        from v_e up to v_free (what the raceline allows here); inf if it never does"""
        if dist <= 0.0:
            return 0.0
        a = self.ego_accel
        t1 = max(v_free - v_e, 0.0) / a                      # end of the acceleration
        gain1 = (v_e - v_o) * t1 + 0.5 * a * t1 ** 2
        if gain1 >= dist:                                    # reached while still accelerating
            b = v_e - v_o
            return (-b + np.sqrt(max(b * b + 2.0 * a * dist, 0.0))) / a
        dv = max(v_free, v_e) - v_o
        return t1 + (dist - gain1) / dv if dv > 1e-3 else np.inf

    def _side_by_side(self, tr, gap, v_e, v_free):
        """stations (s_a, s_b) of the opponent while the cars are side by side, and the time until
        the ego is clear ahead: from gap = overlap_gap to gap = -clear_ahead"""
        t_a = self._time_to_gain(gap - self.overlap_gap, v_e, tr.vs, v_free)
        t_b = self._time_to_gain(gap + self.clear_ahead, v_e, tr.vs, v_free)
        t_a, t_c = min(t_a, 8.0), min(t_b, 8.0)
        return tr.s + tr.vs * t_a - 0.3, tr.s + tr.vs * t_c + 0.3, t_b

    def _pass_line(self, tr, side, room):
        """Frenet d to pass at: lat_sep next to the opponent, less if the wall is closer"""
        sep = min(self.lat_sep, room - 0.5 * CAR_WIDTH - self.wall_margin)
        return tr.d + side * max(sep, self.min_lat_sep - 0.1)

    def _set(self, state, side, target, t, info):
        self.changed = (state != self.state or side != self.side
                        or (state == self.FOLLOW and info.split(' (')[0] != self.info.split(' (')[0]))
        self.state, self.side, self.target, self.info = state, side, target, info

    def ref_offsets(self, s_ref, t_ref):
        """sideways shift [m] of reference points at stations s_ref reached after t_ref seconds:
        full shift from 1.2 m behind to 1.2 m ahead of where the opponent will be then, blended
        to the raceline over the 1.8 m before and after"""
        if self.state != self.PASS or self.target is None:
            return np.zeros(len(s_ref))
        tr = self.target
        g = self.f.wrap(np.asarray(s_ref) - (tr.s + tr.vs * np.asarray(t_ref)))
        w = np.where(g < 0.0, _smoothstep((g + 3.0) / 1.8), 1.0 - _smoothstep((g - 1.2) / 1.8))
        return w * self.d_pass


def predict_opponents(frame, tracks, times, k_max, far=1e6, hold=None):
    """(k_max, len(times), 3) [x, y, yaw] of the moving cars at `times` [s from now]: each keeps
    its speed along the raceline and its sideways offset. Unused rows are far away (no cost).
    hold: {id(track): sideways shift}: these cars are predicted to stay on their offset (no
    sideways speed), moved by the shift -- cars the ego has the right of way against."""
    out = np.full((k_max, len(times), 3), far, dtype=np.float32)
    out[:, :, 2] = 0.0
    times = np.asarray(times, dtype=float)
    for k, tr in enumerate(tracks[:k_max]):
        # it keeps moving sideways for a moment (a defender crossing the track), not for the whole second
        if hold and id(tr) in hold:
            x, y, yaw = frame.pose(tr.s + tr.vs * times, tr.d + hold[id(tr)])
        else:
            x, y, yaw = frame.pose(tr.s + tr.vs * times, tr.d + tr.vd * np.minimum(times, 0.5))
        out[k, :, 0], out[k, :, 1], out[k, :, 2] = x, y, yaw
    return out
