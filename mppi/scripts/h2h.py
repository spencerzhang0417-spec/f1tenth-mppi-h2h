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

    PASS: where the ego is meaningfully faster and there is a straight stretch with room to move
    up NEXT TO the opponent (on either side, the ego may wait for it or change sides while it is
    still behind). It does not have to get past before the straight ends: side by side, the ego on
    the inside of the next corner keeps its line (_hold_inside), on the outside it drops back or
    the switchback takes over. Otherwise FOLLOW: an adaptive-cruise speed cap that keeps
    `follow_gap + follow_time * v` m bumper to bumper.
    """
    FREE, FOLLOW, PASS, SWITCHBACK = 'FREE', 'FOLLOW', 'PASS', 'SWITCHBACK'

    def __init__(self, frame, lat_sep=0.7, min_lat_sep=0.5, wall_margin=0.12, pass_kappa=0.4,
                 min_dv=0.4, look=4.0, follow_gap=0.3, follow_time=0.2, follow_gain=1.5,
                 pass_timeout=8.0, cooldown=1.5, overlap_gap=0.8, clear_ahead=0.6, max_pass_time=5.0,
                 move_kappa=0.6, ego_accel=1.8, follow_decel=1.0, react_time=0.35, follow_step=0.3,
                 corners=(), sb_inside=0.2, sb_max_gap=3.0, sb_speed=0.8, sb_give_up=0.15,
                 sb_commit=False, claim=True, claim_dv=0.2, pass_min_time=0.8, brake_look=2.5,
                 pass_look=3.5, pass_path_kappa=0.35, beside_sep=0.36, drop_dv=0.5, drop_rate=1.2,
                 pass_wait=3.0, beside_gap=0.2, a_lat=4.3, grip_use=0.0, pass_dv=0.8,
                 kappa_inside=0.6, kappa_steer=0.75, slow_pace=0.5):
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
        self.beside_sep, self.drop_dv, self.drop_rate = beside_sep, drop_dv, drop_rate
        self.cap_last = np.inf
        self.pass_wait = pass_wait   # [m] of the opponent's path the ego may wait for room to open
        self.beside_gap = beside_gap  # [m] gap at which the cars count as side by side
        # corners, judged by the speed driven there (see _kappa_limits)
        self.a_lat, self.grip_use, self.pass_dv = a_lat, grip_use, pass_dv
        self.kappa_inside, self.kappa_steer, self.slow_pace = kappa_inside, kappa_steer, slow_pace
        self.vo_f, self.vo_of, self.t_last = 0.0, None, None   # smoothed speed of the target
        self.pace = None
        self.hold = None       # FOLLOW next to a car (pass given up): keep this far ahead [m] beside it
        self.passes = self.passes_given_up = self.side_switches = 0
        self.claim = False     # the ego holds the inside against the target right now
        self.claim_ahead = False
        self.claim_gap = 0.3   # [m] center gap at which the ego counts as next to the car
        self.switched = False  # this pass has changed sides once already
        self.corner = None     # the corner of the running switchback
        self.sb_exit = False   # ... and the ego has cut back to the inside of the defender
        # the two lines of the switchback as the line model drives them (set by the node): offsets
        # d_att / d_def, the defender's speeds v_def and the attacker's v_att, the time at which
        # each waypoint is reached t_att / t_def and the lap times T_att / T_def. None: no model.
        self.sb_model = None
        self.sb_ahead = False  # the ego will be at the crossing of the lines before the defender
        self.sb_lead = np.nan  # ... by this much [m] (the model's prediction, < 0: after it)
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
        self.cap_last = self.v_cap   # of the step before
        self.v_cap = np.inf
        self.changed = False
        self.gap = np.nan if best is None else best_gap
        self.claim = self.claim_ahead = False
        self.hold = None
        # the target's speed along the line, smoothed over ~0.3 s: the estimate is noisy
        # (p95 0.4 m/s), and decisions that predict seconds ahead with it flip with the noise
        dt = 0.0 if self.t_last is None else float(np.clip(t - self.t_last, 0.0, 0.2))
        self.t_last = t
        if best is None or best is not self.vo_of:
            self.vo_of, self.vo_f, self.pace = best, (0.0 if best is None else best.vs), None
        else:
            self.vo_f += dt / (0.3 + dt) * (best.vs - self.vo_f)
        if best is not None:
            # its pace: how fast it drives relative to the raceline's speeds, over the last seconds
            # (taken at one instant out of a corner, while it is still accelerating, the pace
            # came out 10 % low, and the pass was planned against a slower car than it met)
            r = float(np.clip(best.vs / max(f.wps[int(best.s / f.ds) % f.n, 3], 0.5), 0.3, 1.2))
            self.pace = r if self.pace is None else self.pace + dt / (2.0 + dt) * (r - self.pace)
        if self._switchback(s_e, d_e, v_e, best, best_gap, t):
            return
        if best is None:
            self._set(self.FREE, 0, None, t, '')
            return
        if self._hold_inside(s_e, d_e, v_e, best, best_gap, t):
            return
        tr, gap = best, best_gap
        v_free = float(np.mean(f.wps[f.window(s_e, s_e + 6.0), 3]))  # what the ego could drive here
        # what the other car will drive on the stretch ahead of it: as fast, relative to the
        # raceline's speeds, as it is now. Out of a corner it is still accelerating: a pass planned
        # against its corner-exit speed ran out of straight (0.4 m/s of closing where 1.2 was planned)
        v_o = max(self.vo_f, self.pace * float(np.mean(f.wps[f.window(tr.s, tr.s + 6.0), 3])))
        dv = v_free - v_o
        need = self.min_lat_sep + 0.5 * CAR_WIDTH + self.wall_margin  # opponent center -> wall

        # A pass is to get NEXT TO the car where the ego is faster, not to be past it before the
        # straight ends: side by side, the next corner decides (the ego on the inside keeps its
        # line, _hold_inside; on the outside it drops back or the switchback takes over). With
        # "clear ahead before the room ends" as the condition, the ego followed a 0.7x car for
        # whole laps on a track whose longest straight is 10 m.
        # So what needs a straight with room is the stretch where the ego moves up beside the
        # car: from the first overlap (gap = overlap_gap) until the cars are side by side
        # (gap = beside_gap). Where will the opponent be then? Earliest from s_a on (the ego
        # accelerating all the way), for `span` m of its path. On a straight the raceline crosses
        # from one wall to the other, so each side is open only for a part of it: look for the
        # first stretch that is long enough (the ego can wait behind the car for it), on either side.
        # Before the ego gains anything it has to react and to get out from behind the car (in
        # line it may not close in faster than it can brake). Without this time in the plan, a
        # pass with the car's delays was 0.7 s later than planned: its nose got in front 1 m
        # after the room on its side had ended, between the other car and the wall, and it
        # scraped the wall.
        t_out = self.react_time + max(self.beside_sep - abs(d_e - tr.d), 0.0) / 0.6 if gap > self.overlap_gap else 0.0
        t_a = min(t_out + self._time_to_gain(gap - self.overlap_gap, v_e, v_o, v_free), 8.0)
        running = self.state == self.PASS and tr is self.target
        t_b = t_out + self._time_to_gain(gap - self.beside_gap, v_e, v_o, v_free)
        s_a = tr.s + v_o * t_a - 0.6
        span = v_o * (min(t_b, 8.0) - t_a) + 0.9
        search = self.pass_wait if gap > self.overlap_gap else 0.2   # next to it already: no waiting
        ease = 0.07 if running else 0.0    # a running pass is judged a little less strictly
        slack = 0.0 if running else 0.3    # and has to fit with a little to spare to start
        k_o = f.kappa[f.window(tr.s, tr.s + 10.0)]
        k_big = float(k_o[int(np.argmax(np.abs(k_o)))])
        inside = int(np.sign(k_big)) if abs(k_big) > self.pass_kappa else 0   # of the next corner
        kaps = self._kappa_limits(self.pace)
        if running:
            kaps = {sd: k + 0.5 * ease for sd, k in kaps.items()}
        wins = {side: self._open_stretch(tr, side, s_a, span + slack, need - ease, kaps[side], search)
                for side in (1, -1)}

        if running:
            young = t - self.t_start < self.pass_min_time
            win = wins[self.side]
            why = None
            if gap > 0.0 and not young:      # nose ahead: never; just started / switched: not yet
                # (next to the car already, overlapping it: nothing to "get alongside" any more.
                # Out of a switchback the ego is there at the other car's speed; given up, it
                # dropped back a car length that the corner had just gained it.)
                beside_now = gap < CAR_LENGTH + 0.3 and abs(d_e - tr.d) >= self.beside_sep
                if t - self.t_pass > self.pass_timeout or (t_b > self.max_pass_time + 1.0 and not beside_now):
                    why = f'not getting alongside (dv {dv:.1f} m/s)'
                elif win is not None and gap >= self.overlap_gap \
                        and not self._can_finish(win, gap, t_a, v_e, v_o, v_free, 1.0, t_out):
                    why = 'cannot finish before the room ends'
                elif win is None:
                    other = self._open_stretch(tr, -self.side, s_a, span, need,
                                               self._kappa_limits(self.pace)[-self.side], search)
                    # (only from right behind the car and with room to cross: changing sides
                    # from half a meter out, 0.9 m behind it, ended on its rear bumper)
                    if gap >= self.overlap_gap + 0.4 and other is not None and not self.switched \
                            and abs(d_e - tr.d) < 0.2 \
                            and self._can_finish(other, gap, t_a, v_e, v_o, v_free, 0.85, t_out):
                        # (once per pass: left, right, left again is a pass that goes nowhere)
                        self.switched = True
                        # still behind it: cross to the side that is open where the pass will happen
                        self.side_switches += 1
                        self.t_start, win = t, other
                        self._set(self.PASS, -self.side, tr, t,
                                  f'switched to the {"left" if self.side < 0 else "right"}: '
                                  f'room {other[1]:.2f} m, gap {gap:+.1f} m')
                    else:
                        # corner hand-off included: no straight stretch with room left before the
                        # corner. (Next to the car on the inside and faster is _hold_inside above.)
                        why = self._why_closed(tr, self.side, s_a, span, need - ease, kaps[self.side])
            if why is not None:
                self.t_block = t
                self.passes_given_up += 1
                self._set(self.FOLLOW, 0, tr, t, f'pass given up: {why}, gap {gap:+.1f} m')
            else:
                if win is not None:
                    self.d_pass = self._pass_line(tr, self.side, win[1])
                    if win[0] > 0.3 and gap > self.overlap_gap - 0.1:
                        # the open stretch starts win[0] m further on: do not run up to the car's
                        # tail before it gets there. (Loose on purpose: with the smoothed speed
                        # of a car that is still accelerating, a tight cap made the ego late.)
                        self.v_cap = max(tr.vs, self.vo_f) + 0.3 \
                            + max(gap - self.overlap_gap, 0.0) / (win[0] / max(v_o, 0.5))
                if not self.changed:
                    self.info = f'gap {gap:+.1f} m'
                if gap >= self.overlap_gap and not self._can_finish(win, gap, t_a, v_e, v_o, v_free, 1.0, t_out):
                    # (also in the first moments of a pass, when it is not given up yet)
                    self.v_cap = min(self.v_cap, self.vo_f + float(np.sqrt(
                        2.0 * self.follow_decel * max(gap - self.overlap_gap - 0.1, 0.0))))
                if gap > 0.0 and abs(d_e - tr.d) < self.beside_sep:
                    # Still in line behind the car (not out to the side yet, or the wall does not
                    # let it): close in only as fast as the ego can brake again in the gap it
                    # has. Uncapped, it came up to 1 m behind a 0.4x car at +1.3 m/s waiting for
                    # room on a side where the wall was; the room did not come, the pass was
                    # given up, and what was left was full braking and a swerve to the wall.
                    # (the smoothed speed: while the ego pulls out it sees the box from another
                    # side, and the raw estimate of a car driving 1.8 m/s read 0.9 for a moment)
                    v_now = self.vo_f
                    self.v_cap = min(self.v_cap, float(np.sqrt(
                        max(v_now, 0.0) ** 2 + 2.0 * self.follow_decel * max(gap - CAR_LENGTH - 0.1, 0.0))))
        else:
            # the ego's own path: how much it is turning now, and the bends just ahead. Judged by
            # the speed it will have there while it is still behind the car (the car's, and a bit)
            kap_e = float(np.max(np.abs(f.kappa[f.window(s_e, s_e + 1.5)])))
            lim_e = float(np.clip(self.grip_use * self.a_lat / max(v_e + 0.3, 0.5) ** 2,
                                  self.move_kappa, self.kappa_steer))
            idx_p = f.window(s_e, s_e + self.pass_look)
            v_p = np.minimum(f.wps[idx_p, 3], v_o + 0.6)
            lim_p = np.clip(self.grip_use * self.a_lat / np.maximum(v_p, 0.5) ** 2, self.pass_path_kappa, self.kappa_steer)
            over_p = np.abs(f.kappa[idx_p]) >= lim_p
            kap_path = float(np.max(np.abs(f.kappa[idx_p])[over_p])) if over_p.any() else 0.0
            open_sides = [sd for sd in (1, -1) if wins[sd] is not None]
            unfinished = [sd for sd in open_sides if not self._can_finish(wins[sd], gap, t_a, v_e, v_o, v_free, 0.85, t_out)]
            open_sides = [sd for sd in open_sides if sd not in unfinished]
            # the side that opens first; if both do, the one that stays open longer (the ego is
            # often later than planned), the inside of the next corner counting 1.5 m more (side
            # by side there, the corner is the ego's)
            side = min(open_sides, key=lambda sd: (round(wins[sd][0], 1),
                                                   -(wins[sd][2] + (1.5 if sd == inside else 0.0)))) if open_sides else 0
            if dv <= self.min_dv or t_b >= self.max_pass_time:
                why = f'too slow (dv {dv:.1f} m/s)'
            elif gap < self.overlap_gap + 0.1 and abs(d_e - tr.d) < self.beside_sep:
                # right behind its bumper: no room to pull out (a car that had just started to
                # move was "passed" from 0.6 m behind, and touched)
                why = 'too close'
            elif side == 0:
                why = 'no stretch (left: {}, right: {})'.format(
                    *('too short to finish' if sd in unfinished else
                      self._why_closed(tr, sd, s_a, span + slack, need, kaps[sd]) for sd in (1, -1)))
            elif kap_e >= lim_e:
                # the ego itself is still turning: it has no steering left to change lanes
                # (and a move to the inside of a hairpin is tighter than the car can turn)
                why = f'ego in corner (kappa {kap_e:.2f})'
            elif over_p.any():
                # a bend within the next few meters of the ego's own path. Pulling out of line
                # while accelerating out of one corner into the next put the car 0.5 m/s over the
                # speed of that bend and 1.4 m off the raceline: it slid into the wall (twice
                # in two runs, at the right kink after the first hairpin)
                why = f'bend ahead (kappa {kap_path:.2f})'
            elif t - self.t_block <= self.cooldown:
                why = 'cooldown'
            else:
                why = None
            if why is None:
                self.t_pass = self.t_start = t
                self.passes += 1
                self.switched = False
                self.d_pass = self._pass_line(tr, side, wins[side][1])
                self._set(self.PASS, side, tr, t,
                          f'{"left" if side > 0 else "right"}: gap {gap:+.1f} m, dv {dv:.1f} m/s, '
                          f'room {wins[side][1]:.2f} m, opens in {wins[side][0]:.1f} m')
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
            if abs(d_e - tr.d) > self.beside_sep and gap_b < want:
                # Next to the car, not behind it (a pass that was given up): there is nothing in
                # front to run into, so ease off instead of braking for a gap that is not there.
                # Drop back a little slower than the car, staying on this side until behind it.
                # (The law above took the car from 4.0 to 1.9 m/s within half a second here.)
                # The cap comes down at drop_rate, from the speed the car has: a cap a fixed step
                # below the car's own speed is chased by the car at full braking (-2.5 m/s^2).
                last = self.cap_last if np.isfinite(self.cap_last) else v_e
                rate = self.follow_decel if v_e - tr.vs > 0.8 else self.drop_rate   # still closing fast: not so gently
                self.v_cap = max(min(v_o - self.drop_dv, v_soft), min(last, v_e) - rate * dt, 0.0)
                self.hold = max(gap, 0.0) + 1.0

    def _switchback(self, s_e, d_e, v_e, tr, gap, t):
        """the switchback: when the car ahead covers the inside before a corner, take the late-apex
        line through it (wide in, late apex, out on the inside) instead of following it.
        True while it is running: the caller then uses the late-apex line and no speed cap."""
        f = self.f
        if self.state == self.SWITCHBACK:
            c = self.corner
            zone = (c['s_end'] - c['s_split']) % f.length
            if float(f.wrap(s_e - c['s_split'])) > zone:        # the lines are together again: done
                self.corner, self.sb_exit = None, False
                if tr is not None and -2.0 < gap < 1.5 and c['inside'] * (d_e - tr.d) > 0.0:
                    # Not clear of the car yet (next to it, or its nose just behind): the ego keeps
                    # its lane on the inside until it is, as a pass on that side. Both lines end
                    # on the raceline here: handed back to FREE / FOLLOW, the ego steered onto it
                    # with the other car 0.6 m ahead on the same line, and touched its tail.
                    self.t_pass = self.t_start = t
                    self.switched = True
                    self.d_pass = self._pass_line(tr, c['inside'], self.room(tr, c['inside'], tr.s - 0.3, tr.s + 1.5))
                    self._set(self.PASS, c['inside'], tr, t,
                              f'switchback {c["name"]} through, finishing on the inside, gap {gap:+.1f} m')
                else:
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
            self._sb_timing(c, s_e, d_e, v_e, tr, gap)
            # the cut-back: past the turn-in and on the inside of the defender, the ego drives
            # NEXT TO that car (sb_offsets), not along a line that does not know where it is
            self.sb_exit = (tr is not None and not before_commit and -2.0 < gap < 2.5
                            and c['inside'] * (d_e - tr.d) > 0.0)
            clearing = False
            m = self.sb_model
            if m is not None and tr is not None:
                # Crossing behind a car that is on its way to the outside wall: by the time the ego
                # is where that car is now, it is not there any more. Where both will be 0.4 s on,
                # each on its line (the line model), if the car ahead is driving its line.
                io, ie = int(tr.s / f.ds) % f.n, int(s_e / f.ds) % f.n
                err = tr.d - float(m['d_def'][io])
                if abs(err) < 0.4:
                    d_o = float(m['d_def'][int((tr.s + 0.4 * max(tr.vs, 0.0)) / f.ds) % f.n]) + err
                    d_a = float(m['d_att'][int((s_e + 0.4 * max(v_e, 0.0)) / f.ds) % f.n]) + d_e - float(m['d_att'][ie])
                    clearing = c['inside'] * (d_a - d_o) > self.beside_sep + 0.15
            if tr is not None and gap > 0.0 and abs(d_e - tr.d) < self.beside_sep and not clearing:
                # In line behind it (crossing its track from the outside to the inside, or no room
                # to get next to it): close in only as fast as the ego can still brake. An early
                # apex ends in a slow exit; the ego came out of the corner 1.4 m/s faster than
                # the car in front of it.
                v_now = max(min(tr.vs, self.vo_f), 0.0)
                self.v_cap = min(self.v_cap, float(np.sqrt(
                    v_now ** 2 + 2.0 * self.follow_decel * max(gap - CAR_LENGTH - 0.1, 0.0))))
            self.info = f'{c["name"]} gap {gap:+.1f} m' + ('' if np.isnan(self.sb_lead) else
                                                          f', at the crossing {self.sb_lead:+.1f} m')
            return True
        if tr is None or not self.corners:
            return False
        if self.state == self.PASS and (gap < self.overlap_gap or self.side == 0):
            return False    # next to the car already: no cutting back behind it
        for c in self.corners:
            a, b = c['s_decide']
            if (s_e - a) % f.length > (b - a) % f.length:
                continue                                        # not in this corner's decision stretch
            if self.state == self.PASS and self.side == c['inside']:
                continue                                        # already moving up on its inside
            inside_now = c['inside'] * (tr.d + 0.4 * tr.vd)     # where it is heading in 0.4 s
            # (and it has to BE on the inside a little: the sideways speed alone is noisy, and
            # triggered switchbacks behind cars that were 6 cm off the raceline)
            inside_now = min(inside_now, 2.0 * c['inside'] * tr.d)
            v_line = float(f.wps[int(s_e / f.ds) % f.n, 3])
            if 0.3 < gap < self.sb_max_gap and inside_now >= self.sb_inside \
                    and v_e >= self.sb_speed * v_line and v_e >= tr.vs - 0.3:
                self.corner = c
                self.sb_ahead = False
                self.sb_count += 1
                self._set(self.SWITCHBACK, -c['inside'], tr, t,
                          f'{c["name"]}: opponent {c["inside"] * tr.d:+.2f} m inside, gap {gap:+.1f} m, '
                          f'entry speed {v_e:.1f} of {v_line:.1f} m/s')
                return True
        return False

    def _sb_timing(self, c, s_e, d_e, v_e, tr, gap):
        """SWITCHBACK before the lines cross: who is at the crossing first?

        The late-apex line is outside of the defender's on the way in and inside of it on the
        way out, so the two cross once, around the apex. A car that covers the inside brakes
        early and deep for its tight line; the ego, on the outside, brakes later. Driven flat
        out, the ego arrives at the crossing next to that car, which is the one place where it
        must not be: the defender runs wide out of its early apex, onto the line of a car on
        its outside. So the ego is there either clearly before it (then the corner is the ego's:
        it has out-braked the other car around the outside) or behind it, to cut back under it.
        Which one is decided from the line model: both lines' own speed profiles from where
        the cars are now, the defender's scaled to the pace it is driving at.
          before it by more than a car:  nothing to do
          otherwise:                     the late-apex line's speeds scaled down so that the ego
                                         arrives 0.75 m behind the defender, still faster than it
                                         (the run it takes onto the inside out of the corner)"""
        self.sb_lead = np.nan
        m, f = self.sb_model, self.f
        if m is None or tr is None or gap < -1.0 or gap > 4.0:
            return
        side, n = c['inside'], f.n
        if 'i_x' not in c:
            # where the late-apex line gets inside of the defender's
            idx = (int(c['s_commit'] / f.ds) - 20 + np.arange(int(((c['s_end'] - c['s_commit']) % f.length) / f.ds) + 20)) % n
            k = np.nonzero(side * (m['d_att'][idx] - m['d_def'][idx]) >= 0.0)[0]
            c['i_x'] = int(idx[k[0]]) if len(k) else int(c['s_apex'] / f.ds) % n
        ix, ie, io = c['i_x'], int(s_e / f.ds) % n, int(tr.s / f.ds) % n
        if float(f.wrap(ix * f.ds - s_e)) <= 0.0 or side * (d_e - tr.d) > 0.0:
            return                                              # past the crossing, or inside already
        te = float((m['t_att'][ix] - m['t_att'][ie]) % m['T_att'])
        te *= float(np.clip(m['v_att'][ie] / max(v_e, 0.5), 1.0, 1.3))   # (slower than the line now: later)
        if float(f.wrap(ix * f.ds - tr.s)) > 0.0:
            to = float((m['t_def'][ix] - m['t_def'][io]) % m['T_def'])
        else:
            to = -float((m['t_def'][io] - m['t_def'][ix]) % m['T_def'])   # the defender is through
        to *= float(np.clip(m['v_def'][io] / max(self.vo_f, 0.5), 0.75, 1.4))
        v_x = float(m['v_def'][ix])
        self.sb_lead = (to - te) * v_x
        # (hysteresis: a decision that flips at every step is braking and accelerating in turns)
        self.sb_ahead = self.sb_lead > (1.0 if self.sb_ahead else 1.5)
        if not self.sb_ahead:
            t_need = to + 0.75 / max(v_x, 0.5)
            scale = float(np.clip(te / max(t_need, 0.05), 0.5, 1.0))
            self.v_cap = min(self.v_cap, max(float(m['v_att'][ie]) * scale, 0.8 * max(self.vo_f, 0.0)))

    def sb_predict(self, times):
        """SWITCHBACK: where the defender will be at `times` [s from now] if it drives the
        defender's line of this corner (the line model), (s, d) arrays; None if it is not on
        that line (more than 0.5 m off it) or there is no model. Its early apex takes it from
        the inside wall to the outside one within a second: predicted on the offset it has now,
        it is in the way of the cut-back and gone by the time the ego gets there."""
        m, tr, f = self.sb_model, self.target, self.f
        if self.state != self.SWITCHBACK or m is None or tr is None:
            return None
        io = int(tr.s / f.ds) % f.n
        err = tr.d - float(m['d_def'][io])
        if abs(err) > 0.5:
            return None
        s, d, ratio = np.empty(len(times)), np.empty(len(times)), max(tr.vs, 0.0) / max(float(m['v_def'][io]), 0.5)
        ratio = float(np.clip(ratio, 0.6, 1.4))
        s_k, t_k = tr.s, 0.0
        for k, t in enumerate(times):
            i = int(s_k / f.ds) % f.n
            s_k += float(m['v_def'][i]) * ratio * (t - t_k)
            t_k = t
            s[k], d[k] = s_k, float(m['d_def'][int(s_k / f.ds) % f.n]) + err
        return s % f.length, d

    def sb_offsets(self, o_att, s_e, d_e, v):
        """SWITCHBACK after the turn-in: the late-apex line (o_att, its offsets from the raceline),
        moved further to the inside where the defender will be when the ego gets there, so that
        the ego comes out of the corner lat_sep next to that car and not behind it. The late-apex
        line and the defender's line both end on the raceline; next to each other the cars cannot.
        None: the late-apex line as it is."""
        if self.state != self.SWITCHBACK or not self.sb_exit or self.target is None:
            return None
        f, side, tr = self.f, self.corner['inside'], self.target
        s_wp = np.arange(f.n) * f.ds
        ahead = np.maximum(f.wrap(s_wp - s_e), 0.0)
        g = f.wrap(s_wp - (tr.s + tr.vs * ahead / max(v, 1.0)))     # station relative to the car, then
        k = max(1.0, v / 4.0)
        w = np.where(g < 0.0, _smoothstep((g + 1.2 + 1.8 * k) / (1.8 * k)), 1.0 - _smoothstep((g - 1.6) / (3.0 * k)))
        # A defender coming back from the outside is moving towards the ego's lane: where will it
        # be in 0.4 s? On its line (the line model), if it is driving that; else by its sideways
        # speed now. (By its sideways speed alone, a car returning to the raceline at 1 m/s was
        # put 0.45 m further in than it ever came, and the ego's lane against the inside wall.)
        m = self.sb_model
        io = int(tr.s / f.ds) % f.n
        if m is not None and abs(tr.d - float(m['d_def'][io])) < 0.5:
            d_next = float(m['d_def'][int((tr.s + 0.4 * max(tr.vs, 0.0)) / f.ds) % f.n]) + tr.d - float(m['d_def'][io])
        else:
            d_next = tr.d + 0.4 * tr.vd
        # (a car's width and a hand: next to it out of a corner, not lat_sep as on a straight)
        lane = side * max(side * tr.d, side * d_next) + side * (self.min_lat_sep + 0.05)
        extra = side * np.maximum(side * (lane - o_att), 0.0) * w
        if float(np.max(np.abs(extra))) < 0.05:
            # the line is far enough from the car as it is: drive it as it is (re-shaped from the
            # car's own position it gets bends of its own, 0.4 m/s out of the corner for nothing)
            return None
        # from where the car is relative to the line, sideways no faster than it can follow
        extra = self.shape(extra, s_e, d_e - o_att[int(s_e / f.ds) % f.n], v=v)
        return self.clamp(o_att + extra)

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
        # Really next to it: half a car of overlap and the bodies clear of each other. (With
        # "within 0.8 m behind and 0.25 m to the side" the ego claimed the corner from behind the
        # other car's inside rear wheel; the car turned in to its apex and they touched.)
        beside = gap < self.claim_gap and abs(d_e - tr.d) > self.beside_sep
        if not (beside and side == inside and (v_e > tr.vs + self.claim_dv or gap < 0.0)):
            return False
        self.claim = True
        self.claim_ahead = gap < 0.0     # nose in front: the other car is the one to leave room
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

    def _kappa_limits(self, pace):
        """how much curvature of the raceline the cars may be side by side on, per waypoint and
        side {+1: left, -1: right}.

        The fixed limit (pass_kappa) is right for two cars at the limit of grip. Behind a slow car
        nobody is: next to a 0.6x car the ego needs a third of the grip the raceline's speed there
        uses, and it still followed it down a straight with a meter of room on both sides because
        the stretch ended in a bend of kappa 0.46. So: the speed the ego passes at there (the
        other car's, pace x the raceline's, plus pass_dv) may use grip_use of the lateral
        acceleration a_lat. On the inside of a corner up to kappa_inside (the line next to the
        car is tighter than the raceline, and the car turns no tighter than it can steer): moving
        up on the inside into a corner is the pass, the corner is then the ego's (_hold_inside).
        On the outside only around a car slower than slow_pace.

        grip_use = 0 (the default) turns all of this off: pass_kappa everywhere. With 0.7 the ego
        did pass a 0.6x car on the inside into the first hairpin, and in the same tests a pass
        into a corner was given up 0.7 m behind a 0.5x car at +1 m/s: full braking, a swerve, the
        wall. Not before giving a pass up is safe at any moment."""
        f = self.f
        v_line = f.wps[:, 3]
        v_pass = np.minimum(v_line, pace * v_line + self.pass_dv)
        grip = self.grip_use * self.a_lat / np.maximum(v_pass, 0.5) ** 2
        inside = np.sign(f.kappa)
        out = {}
        for side in (1, -1):
            cap = np.where(inside == side, self.kappa_inside, 1.0 if pace < self.slow_pace else self.pass_kappa)
            out[side] = np.maximum(self.pass_kappa, np.minimum(cap, grip))
        return out

    def _open_stretch(self, tr, side, s_from, span, need, kap_max, search):
        """the first `span` m of the opponent's path, starting at station s_from or up to `search` m
        after it, with `need` m of room on `side` of the car and no corner (|kappa| < kap_max, a
        value per waypoint).
        Returns (how much later than s_from it starts [m], smallest room in it [m], how long
        the open run goes on from there [m], looking 6 m past the stretch, and whether what ends
        the run is a corner with `side` as its inside) or None."""
        f = self.f
        idx = f.window(s_from, s_from + search + span + 6.0)
        w = (f.w_left[idx] if side > 0 else f.w_right[idx]) - side * tr.d
        no_room, corner = w < need, np.abs(f.kappa[idx]) >= kap_max[idx]
        bad = np.r_[0, np.cumsum(no_room | corner)]
        m = min(int(np.ceil(span / f.ds)) + 1, len(idx))
        free = np.nonzero(bad[m:] - bad[:-m] == 0)[0]
        free = free[free * f.ds <= search + 1e-6]
        if len(free) == 0:
            return None
        i = int(free[0])
        end = np.nonzero(bad[i + 1:] > bad[i])[0]              # where the open run ends
        j = int(end[0]) if len(end) else len(idx)              # first point that is not open
        inside_end = bool(j < len(idx) and corner[i + j] and not no_room[i + j]
                          and np.sign(f.kappa[idx[i + j]]) == side)
        return i * f.ds, float(np.min(w[i:i + m])), j * f.ds if len(end) else 99.0, inside_end

    def _can_finish(self, win, gap, t_a, v_e, v_o, v_free, slack, t_out=0.0):
        """A pass has to END somewhere safe: can the ego be there before the open stretch (win,
        of _open_stretch) is over? Safe is: its nose in front of the other car (then the place
        is taken), or, where the stretch ends at a corner with the ego on the inside, really next
        to the car (then the corner is the ego's, _hold_inside). Anywhere else next to the car
        when the room or the straight runs out leaves only full braking and a swerve: that is
        how the car ended against the wall twice."""
        if win is None:
            return False
        delay, _, run, inside_end = win
        t_end = t_a + (delay + run - 0.6) / max(v_o, 0.3)          # the opponent at the end of the run
        g_ok = self.claim_gap - 0.05 if inside_end else -0.3
        return t_out + self._time_to_gain(gap - g_ok, v_e, v_o, v_free) <= slack * t_end

    def _why_closed(self, tr, side, s_from, span, need, kap_max):
        """what is in the way of a pass on `side` over the stretch from s_from: 'room 0.42 m',
        'corner 0.94' (the curvature over the limit there) or both"""
        f = self.f
        idx = f.window(s_from, s_from + span)
        room = float(np.min((f.w_left[idx] if side > 0 else f.w_right[idx]) - side * tr.d))
        over = np.abs(f.kappa[idx]) >= kap_max[idx]
        why = ([f'room {room:.2f} m'] if room < need else []) \
            + ([f'corner {float(np.max(np.abs(f.kappa[idx])[over])):.2f}'] if over.any() else [])
        return ' + '.join(why) if why else 'later only'

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

    def _pass_line(self, tr, side, room):
        """Frenet d to pass at: lat_sep next to the opponent, less if the wall is closer"""
        sep = min(self.lat_sep, room - 0.5 * CAR_WIDTH - self.wall_margin)
        return tr.d + side * max(sep, self.min_lat_sep - 0.1)

    def _set(self, state, side, target, t, info):
        self.changed = (state != self.state or side != self.side
                        or (state == self.FOLLOW and info.split(' (')[0] != self.info.split(' (')[0]))
        self.state, self.side, self.target, self.info = state, side, target, info

    def ref_offsets(self, s_ref, t_ref, v=4.0):
        """sideways shift [m] of reference points at stations s_ref reached after t_ref seconds:
        full shift from 1.2 m behind to 1.2 m ahead of where the opponent will be then, blended
        to the raceline over the 1.8 m before and after"""
        if self.state != self.PASS or self.target is None:
            return np.zeros(len(s_ref))
        tr = self.target
        g = self.f.wrap(np.asarray(s_ref) - (tr.s + tr.vs * np.asarray(t_ref)))
        k = max(1.0, v / 4.0)      # the blends as long in time at any speed (they were made at 4 m/s)
        w = np.where(g < 0.0, _smoothstep((g + 1.2 + 1.8 * k) / (1.8 * k)), 1.0 - _smoothstep((g - 1.6) / (3.0 * k)))
        return self.clamp(w * self.d_pass, (np.asarray(s_ref) / self.f.ds).astype(int) % self.f.n)

    def shape(self, o, s_e, d_e, slope=0.2, soften=1.5, v=None):
        """offsets of a passing line (one per waypoint) made drivable at speed: the line leaves
        the car's own sideways position d_e, and moves sideways by at most `slope` m per m in
        either direction -- it starts to come back early enough for where it has to be back (in
        front of the other car, or where the wall closes in) -- with the corners rounded over
        `soften` m. The sideways move of 0.6 m used to come back within 1.8 m: a bend of its own
        that the car may take at 2.1 m/s, slower than a 0.7x opponent drives on the straight."""
        f = self.f
        if v is not None:
            # about 0.7 m/s sideways: 0.2 m per m at 3.5 m/s (more than that the tires do not hold
            # while accelerating), up to 0.35 behind a slow car, where 0.2 took 1.3 s to get out
            # (0.7 m/s sideways at any speed: with 0.2 m per m as the least, the line moved over at
            # 1.2 m/s at 6 m/s, a bend its own speed profile takes at 4.3 m/s, on a straight)
            slope = float(np.clip(0.7 / max(v, 0.5), 0.1, 0.35))
            soften = soften * max(1.0, v / 4.0)
        n, i0 = f.n, int(s_e / f.ds) % f.n
        m = n // 2                                    # the half of the lap ahead of the car
        u = np.roll(np.asarray(o, dtype=float), -i0)[:m]
        sign = 1.0 if u[int(np.argmax(np.abs(u)))] >= 0.0 else -1.0
        u, k, a = sign * u, slope * f.ds * np.arange(m), sign * d_e
        u = np.minimum.accumulate(u - k) + k                       # rises by at most `slope`
        u = (np.minimum.accumulate((u + k)[::-1])[::-1]) - k       # and falls by at most `slope`
        u = np.clip(u, a - k, a + k)                               # from where the car is
        # rounded corners. Before the car the line continues straight (padded with the car's own
        # position): a corner AT the car is a bend the speed profile brakes for, on every step
        w = max(1, int(0.5 * soften / f.ds))
        u = np.convolve(np.r_[np.full(w, a), u, np.full(w, u[-1])], np.ones(2 * w + 1) / (2 * w + 1), mode='valid')
        out = np.zeros(n)
        out[:m] = sign * u
        # behind the car: its own position for 2 m, then back to the raceline over 3 m (what is
        # behind is not driven, but a jump there is a bend for the speed profile just as well)
        back = f.ds * np.arange(1, n - m + 1)[::-1]                # distance behind, per point
        out[m:] = d_e * (1.0 - _smoothstep((back - 2.0) / 3.0))
        return self.clamp(np.roll(out, i0))

    def clamp(self, o, idx=None):
        """sideways offsets (one per waypoint, or at waypoint indices idx) kept inside the track:
        the car's side wall_margin off the wall at every station. Without it the passing line
        went 0.48 m to the right where the wall is 0.5 m from the raceline; the car would not
        follow it there, stayed behind the opponent and lost the straight."""
        f = self.f
        hi = np.maximum(f.w_left - 0.5 * CAR_WIDTH - self.wall_margin, 0.0)
        lo = -np.maximum(f.w_right - 0.5 * CAR_WIDTH - self.wall_margin, 0.0)
        if idx is not None:
            hi, lo = hi[idx], lo[idx]
        return np.clip(o, lo, hi)


def predict_opponents(frame, tracks, times, k_max, far=1e6, hold=None, margin=0.65, margins=None):
    """(k_max, len(times), 4) [x, y, yaw, soft margin] of the moving cars at `times` [s from
    now]: each keeps its speed along the raceline and its sideways offset. Unused rows are far
    away (no cost).
    hold: {id(track): sideways shift}: these cars are predicted to stay on their offset (no
    sideways speed), moved by the shift -- cars the ego has the right of way against.
    margin: [m] distance at which the soft cost of a car starts; margins: {id(track): margin}
    for single cars (smaller for the car being passed: next to it is where the ego wants to be)."""
    out = np.full((k_max, len(times), 4), far, dtype=np.float32)
    out[:, :, 2] = 0.0
    out[:, :, 3] = margin
    times = np.asarray(times, dtype=float)
    for k, tr in enumerate(tracks[:k_max]):
        # it keeps moving sideways for a moment (a defender crossing the track), not for the whole second
        if hold and id(tr) in hold:
            x, y, yaw = frame.pose(tr.s + tr.vs * times, tr.d + hold[id(tr)])
        else:
            x, y, yaw = frame.pose(tr.s + tr.vs * times, tr.d + tr.vd * np.minimum(times, 0.5))
        out[k, :, 0], out[k, :, 1], out[k, :, 2] = x, y, yaw
        if margins and id(tr) in margins:
            out[k, :, 3] = margins[id(tr)]
    return out
