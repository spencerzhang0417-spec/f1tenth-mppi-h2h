#!/usr/bin/env python3
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from utils import calc_interpolated_ref_trajectory

# TODO CHECK: include needed ROS msg type headers and libraries
import dataclasses
import os
import time
import re
from rclpy.qos import QoSProfile
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


@dataclass(frozen=True)
class MPPIConfig:
    # frozen so jax.jit can take it as a static argument: use tuples, not arrays
    NX: int = 4  # length of state vector: x = [p_x, p_y, yaw, v]
    NU: int = 2  # length of input vector: u = [a, delta]
    T: int = 1024  # number of sampled control sequences
    H: int = 20  # horizon length
    DT: float = 0.05  # time step [s]
    INTEGRATOR: str = "euler"  # "euler" or "rk4" (bonus)

    # ---------------------------------------------------
    # TODO: you may need to tune the following parameters
    LAMBDA: float = 0.01  # temperature
    U_STD: tuple = (1.0, 0.1)  # std of the sampled inputs [a, delta]
    Q: tuple = (5.0, 5.0, 1.0, 1.0)  # state error cost [x, y, yaw, v]
    Qf: tuple = (5.0, 5.0, 1.0, 1.0)  # final state error cost [x, y, yaw, v]
    R: tuple = (0.01, 0.01)  # input cost [a, delta]
    Rd: tuple = (0.01, 1.0)  # input difference cost [a, delta]
    # ---------------------------------------------------

    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # minimum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MIN_ACCEL: float = -3.0  # minimum acceleration [m/ss]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]
    MIN_SPEED: float = 0.0  # minimum speed [m/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]

    # ---------------- added options (not in the skeleton) ----------------
    # extra cost terms (README III.b: constraints enforced as penalties)
    SPEED_PENALTY: float = 50.0     # weight on (v - MAX_SPEED)^2 and (MIN_SPEED - v)^2
    OFFTRACK_DIST: float = 0.6      # [m] allowed lateral distance from the reference path
    OFFTRACK_PENALTY: float = 50.0  # cost per step beyond OFFTRACK_DIST (0 disables)
    # Bonus: proposal distribution. "clip" = Gaussian + clipping (README default),
    #        "truncnorm" = Gaussian truncated to the input bounds
    SAMPLING: str = "clip"
    # Bonus: covariance update (README IV.5). When True the node uses mppi_step_cov()
    ADAPTIVE_COV: bool = False
    COV_ALPHA: float = 0.3            # blend: Sigma <- (1-a) Sigma_prior + a Sigma_weighted
    COV_MIN_STD: tuple = (0.2, 0.02)  # floor on the std so the proposal never collapses
    # MPPI iterations per odom message: N_ITER_COLD right after a (re)start from U = 0,
    # N_ITER afterwards (the warm start already carries the previous solution)
    N_ITER_COLD: int = 10
    N_ITER: int = 1


def dynamics(x, u, cfg):
    """
    Kinematic bicycle model f(x, u) of a single state (README Section II.a)
    Use jnp instead of np or math so JAX can trace it.
    :param x: state [p_x, p_y, yaw, v], shape (NX,)
    :param u: input [a, delta], shape (NU,)
    :param cfg: MPPIConfig, cfg.WB is the wheelbase
    :return: dx/dt, shape (NX,)
    """
    # TODO: fill in the kinematic model
    yaw, v = x[2], x[3]
    a, delta = u[0], u[1]
    return jnp.stack([
        v * jnp.cos(yaw),
        v * jnp.sin(yaw),
        v * jnp.tan(delta) / cfg.WB,
        a,
    ])


def step_euler(x, u, cfg):
    """
    Forward Euler step: x_{k+1} = x_k + f(x_k, u_k) * DT (README Section II.b.i)
    :return: next state, shape (NX,)
    """
    # TODO: fill in the Euler step
    return x + dynamics(x, u, cfg) * cfg.DT


def step_rk4(x, u, cfg):
    """
    Bonus: Runge-Kutta 4 step (README Section II.b.ii)
    :return: next state, shape (NX,)
    """
    # TODO: fill in the RK4 step
    dt = cfg.DT
    k1 = dynamics(x, u, cfg)
    k2 = dynamics(x + 0.5 * dt * k1, u, cfg)
    k3 = dynamics(x + 0.5 * dt * k2, u, cfg)
    k4 = dynamics(x + dt * k3, u, cfg)
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def step(x, u, cfg):
    """
    Discretized dynamics F(x, u): the state after DT, holding u constant
    """
    if cfg.INTEGRATOR == "rk4":
        return step_rk4(x, u, cfg)
    return step_euler(x, u, cfg)


def simulate(x0, U, cfg):
    """
    Simulate one control sequence from x0
    :param x0: initial state, shape (NX,)
    :param U: control sequence u_0 ... u_{H-1}, shape (H, NU)
    :return: states x_0 ... x_H, shape (H+1, NX), starting with x0
    """
    # TODO: apply step() H times. A for loop works, but jax.lax.scan compiles
    #       much faster: jax.lax.scan(f, x0, U) calls carry, out = f(carry, u)
    #       for each row u of U and returns the last carry and the stacked outs
    x0 = jnp.asarray(x0, dtype=jnp.float32)

    def f(x, u):
        x_next = step(x, u, cfg)
        return x_next, x_next

    _, xs = jax.lax.scan(f, x0, U)
    return jnp.concatenate([x0[None, :], xs], axis=0)


def rollout(x0, U, cfg):
    """
    Simulate all T sampled control sequences from x0, in parallel
    :param x0: initial state, shape (NX,)
    :param U: sampled control sequences, shape (T, H, NU)
    :return: states, shape (T, H+1, NX)
    """
    # TODO: vectorize simulate() over the first axis of U with jax.vmap
    #       (its in_axes say which axis of each argument to map over, None for none)
    # cfg is a static Python object, so it is closed over instead of passed to vmap
    return jax.vmap(lambda Uk: simulate(x0, Uk, cfg), in_axes=0)(U)


def cost(X, U, ref, cfg):
    """
    Cost of each step of one trajectory (README Section III.a): c[t] is the cost
    of applying u_t and reaching x_{t+1}
        c[t] = (x_{t+1} - ref_{t+1})^T Q (x_{t+1} - ref_{t+1})   (Qf for the last step)
             + u_t^T R u_t
             + (u_t - u_{t-1})^T Rd (u_t - u_{t-1})              (from t = 1)
    :param X: states x_0 ... x_H, shape (H+1, NX)
    :param U: inputs u_0 ... u_{H-1}, shape (H, NU)
    :param ref: reference states, shape (H+1, NX)
    :return: c, shape (H,)
    """
    # TODO: Q, Qf, R and Rd are diagonal: e^T Q e = sum(jnp.array(cfg.Q) * e**2)
    # TODO: wrap the heading error to [-pi, pi]
    # TODO: add your own penalties if you want, e.g. for going over cfg.MAX_SPEED
    H = U.shape[0]
    Q = jnp.array(cfg.Q)
    Qf = jnp.array(cfg.Qf)
    R = jnp.array(cfg.R)
    Rd = jnp.array(cfg.Rd)

    # state error of x_1 ... x_H, heading error wrapped to [-pi, pi]
    e = X[1:] - ref[1:]
    e = e.at[:, 2].set(jnp.arctan2(jnp.sin(e[:, 2]), jnp.cos(e[:, 2])))
    W = jnp.tile(Q, (H, 1)).at[H - 1].set(Qf)  # Q for t < H-1, Qf for the last step
    c = jnp.sum(W * e ** 2, axis=1)

    # input cost
    c = c + jnp.sum(R * U ** 2, axis=1)

    # input difference cost, from t = 1
    dU = U[1:] - U[:-1]
    c = c.at[1:].add(jnp.sum(Rd * dU ** 2, axis=1))

    # extra penalty: speed outside [MIN_SPEED, MAX_SPEED]
    v = X[1:, 3]
    c = c + cfg.SPEED_PENALTY * (jnp.maximum(v - cfg.MAX_SPEED, 0.0) ** 2
                                 + jnp.maximum(cfg.MIN_SPEED - v, 0.0) ** 2)

    # extra penalty: leaving the corridor around the reference path (keeps the car off the walls).
    # distance from each predicted position to the closest reference point
    d = jnp.min(jnp.linalg.norm(X[1:, None, :2] - ref[None, :, :2], axis=-1), axis=1)
    c = c + cfg.OFFTRACK_PENALTY * (d > cfg.OFFTRACK_DIST)
    return c


@partial(jax.jit, static_argnames=("cfg",))
def mppi_step(key, U, x0, ref, cfg):
    """
    One MPPI update (README Section IV)
    :param key: JAX random key, a new one at every call
    :param U: mean control sequence, already warm started, shape (H, NU)
    :param x0: current state, shape (NX,)
    :param ref: reference states, shape (H+1, NX)
    :param cfg: MPPIConfig
    :return: the new mean control sequence, shape (H, NU),
             the sampled trajectories (for visualization), shape (T, H+1, NX),
             and the weights of the samples at each step, shape (T, H)
    For the covariance update (bonus), you may change the inputs and outputs.
    """
    # --------------------------------------------------------
    # TODO 1. Sample: T control sequences around U with noise of std cfg.U_STD
    #         (jax.random.normal(key, shape)), clipped to the input bounds.
    #         Bonus: sample from a truncated distribution instead of clipping

    # TODO 2. Simulate: roll out all T sequences from x0

    # TODO 3. Score: the cost of each step of each sample, c of shape (T, H)
    #         (vectorize cost() with jax.vmap), then the cost-to-go
    #         S[k, t] = c[k, t] + ... + c[k, H-1] (cumsum of the reversed steps)

    # TODO 4. Weight: for each step t, normalize S[:, t] to [0, 1] over the
    #         samples, (S - min) / (max - min + 1e-6), then take its softmax
    #         over the samples, w[:, t] = softmax(-S[:, t] / cfg.LAMBDA)

    # TODO 5. Update: U_new[t] = sum over the samples k of w[k, t] * u_k[t]
    #         Bonus: also update the covariance
    # --------------------------------------------------------
    # Steps 1-5 live in _mppi_update() so the covariance bonus (mppi_step_cov)
    # can reuse them; mppi_step keeps the skeleton's inputs and outputs.
    U_new, X, w, _ = _mppi_update(key, U, x0, ref, _prior_sigma(cfg), cfg)
    return U_new, X, w


# ---------------------------------------------------------------------------
# Added helpers for mppi_step (sampling, update, covariance bonus)
# ---------------------------------------------------------------------------
def _bounds(cfg):
    lo = jnp.array([cfg.MIN_ACCEL, cfg.MIN_STEER])
    hi = jnp.array([cfg.MAX_ACCEL, cfg.MAX_STEER])
    return lo, hi


def _prior_sigma(cfg):
    """Fixed covariance diag(U_STD^2) for every step, shape (H, NU, NU)"""
    return jnp.tile(jnp.diag(jnp.array(cfg.U_STD) ** 2)[None], (cfg.H, 1, 1))


def _sample(key, U, Sigma, cfg):
    """Step 1: T control sequences around U, shape (T, H, NU), inside the input bounds"""
    lo, hi = _bounds(cfg)
    U = jnp.clip(U, lo, hi)
    shape = (cfg.T, cfg.H, cfg.NU)
    if cfg.SAMPLING == "truncnorm":
        # Bonus: Gaussian truncated to [lo, hi] (per-channel std = sqrt(diag Sigma))
        std = jnp.maximum(jnp.sqrt(jnp.diagonal(Sigma, axis1=1, axis2=2)), 1e-6)  # (H, NU)
        z = jax.random.truncated_normal(key, (lo - U) / std, (hi - U) / std, shape)
        u = U + std * z
    else:
        # Gaussian N(U, Sigma_t) (Sigma_t may be full with the covariance bonus), then clip
        L = jnp.linalg.cholesky(Sigma + 1e-9 * jnp.eye(cfg.NU))  # (H, NU, NU)
        z = jax.random.normal(key, shape)
        u = U + jnp.einsum('hij,thj->thi', L, z)
    return jnp.clip(u, lo, hi)  # also guards against round-off for truncnorm


def _mppi_update(key, U, x0, ref, Sigma, cfg):
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    ref = jnp.asarray(ref, dtype=jnp.float32)

    # 1. Sample
    u = _sample(key, U, Sigma, cfg)                                   # (T, H, NU)
    # 2. Simulate
    X = rollout(x0, u, cfg)                                           # (T, H+1, NX)
    # 3. Score: per-step cost, then cost-to-go
    c = jax.vmap(lambda Xk, uk: cost(Xk, uk, ref, cfg))(X, u)         # (T, H)
    S = jnp.flip(jnp.cumsum(jnp.flip(c, axis=1), axis=1), axis=1)     # S[k,t] = sum_{t'>=t} c[k,t']
    # 4. Weight: normalize each step to [0, 1] over the samples, softmax over the samples
    S_min = jnp.min(S, axis=0, keepdims=True)
    S_max = jnp.max(S, axis=0, keepdims=True)
    S_n = (S - S_min) / (S_max - S_min + 1e-6)
    w = jax.nn.softmax(-S_n / cfg.LAMBDA, axis=0)                     # (T, H), sums to 1 over T
    # 5. Update: weighted average of the samples (stays inside the bounds by construction)
    U_new = jnp.einsum('th,thu->hu', w, u)                            # (H, NU)
    # Bonus: weighted covariance  Sigma_t = sum_k w_t^k (u_t^k - U_t)(u_t^k - U_t)^T
    du = u - U_new[None]
    Sigma_w = jnp.einsum('th,thi,thj->hij', w, du, du)                # (H, NU, NU)
    return U_new, X, w, Sigma_w


@partial(jax.jit, static_argnames=("cfg",))
def mppi_step_cov(key, U, Sigma, x0, ref, cfg):
    """
    Bonus: MPPI update that also adapts the proposal covariance (README IV.5).
    :param Sigma: proposal covariance per step, shape (H, NU, NU), already warm started
    :return: U_new, X, w (as mppi_step) and the new covariance, shape (H, NU, NU)
    """
    U_new, X, w, Sigma_w = _mppi_update(key, U, x0, ref, Sigma, cfg)
    # blend with the prior and keep a minimum std so exploration never dies out
    Sigma_new = (1.0 - cfg.COV_ALPHA) * _prior_sigma(cfg) + cfg.COV_ALPHA * Sigma_w
    Sigma_new = Sigma_new + jnp.diag(jnp.array(cfg.COV_MIN_STD) ** 2)[None]
    return U_new, X, w, Sigma_new


_simulate_jit = jax.jit(simulate, static_argnames=("cfg",))  # for visualization only


# ---------------------------------------------------------------------------
# Added: waypoint loading
# ---------------------------------------------------------------------------
# Waypoints are read from mppi/waypoints/<waypoint_file> (default: <track>.csv, the same
# naming as the MPC lab, so its levine.csv / spielberg.csv can be copied over as is).
# Columns are detected from the header comment, otherwise:
#   >= 4 columns  -> x, y, yaw, v   (MPC-lab format; yaw is recomputed from x, y)
#   2-3 columns   -> x, y           (speed profile built from curvature)
#   f1tenth centerline "# x_m, y_m, w_tr_right_m, w_tr_left_m"             -> x, y (curvature speed)
#   f1tenth raceline   "# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax"  -> x_m, y_m, vx_mps
# If <track>.csv is missing, the simulator centerline below is used as a fallback.
WAYPOINTS = {
    'levine': dict(fallback='levine_centerline.csv',
                   ccw=True,   # graded run: counter-clockwise
                   v_min=1.5, v_max=4.0, a_lat=3.0, a_lon=3.0),
    'spielberg': dict(fallback='Spielberg_centerline.csv',
                      ccw=None,  # graded run: direction of the centerline
                      v_min=2.0, v_max=6.0, a_lat=4.0, a_lon=3.0),
}
WAYPOINT_SPACING = 0.1  # [m] utils.calc_interpolated_ref_trajectory assumes uniform spacing


def _read_csv(path):
    """numeric rows of a csv (',' ';' or whitespace separated) and its last '#' header line"""
    rows, header = [], ''
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                # keep the column-name line (e.g. "x_m, y_m, ..."), not a later free-text comment
                if not re.search(r'\bx_m\b', header):
                    header = line.lstrip('#').strip()
                continue
            if not line:
                continue
            try:
                rows.append([float(s) for s in re.split(r'[,;\s]+', line) if s])
            except ValueError:
                continue  # header line
    n = min(len(r) for r in rows)
    return np.array([r[:n] for r in rows]), header


def detect_columns(path):
    """(x_col, y_col, v_col) of a waypoint csv; v_col None -> speed from curvature"""
    data, header = _read_csv(path)
    names = [h.strip().lower() for h in re.split(r'[,;]', header)]
    if 'x_m' in names and 'y_m' in names:  # f1tenth track files
        v_col = names.index('vx_mps') if 'vx_mps' in names else None
        return names.index('x_m'), names.index('y_m'), v_col
    return 0, 1, (3 if data.shape[1] >= 4 else None)


def load_waypoints(path, x_col=0, y_col=1, v_col=None, ccw=None,
                   v_min=1.5, v_max=6.0, a_lat=3.0, a_lon=3.0, ds=WAYPOINT_SPACING, **_):
    """
    Load a closed track, resample it uniformly, fix its direction and give it yaw and speed.
    :return: (N, 4) array of [x, y, yaw, v]
    """
    data, _ = _read_csv(path)
    xy = data[:, [x_col, y_col]]
    v = data[:, v_col] if v_col is not None else None

    # drop duplicated points (nearest_point divides by segment length)
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]
    xy = xy[keep]
    v = v[keep] if v is not None else None
    if np.linalg.norm(xy[0] - xy[-1]) < 1e-3:  # closing point
        xy = xy[:-1]
        v = v[:-1] if v is not None else None

    # resample uniformly along the closed loop
    xy_c = np.vstack([xy, xy[:1]])
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy_c, axis=0), axis=1))]
    n = int(s[-1] / ds)
    s_new = np.linspace(0.0, s[-1], n, endpoint=False)
    x = np.interp(s_new, s, xy_c[:, 0])
    y = np.interp(s_new, s, xy_c[:, 1])
    if v is not None:
        v = np.interp(s_new, s, np.r_[v, v[:1]])
    ds_real = s[-1] / n

    # driving direction (shoelace: positive area = counter-clockwise)
    if ccw is not None:
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        if (area > 0) != ccw:
            x, y = x[::-1].copy(), y[::-1].copy()
            v = v[::-1].copy() if v is not None else None

    # heading from central differences
    yaw = np.arctan2(np.roll(y, -1) - np.roll(y, 1), np.roll(x, -1) - np.roll(x, 1))

    if v is None:
        # speed profile from curvature: v = sqrt(a_lat / |kappa|), then accel / decel limits
        dyaw = np.arctan2(np.sin(np.roll(yaw, -1) - yaw), np.cos(np.roll(yaw, -1) - yaw))
        kappa = dyaw / ds_real
        win = max(1, int(1.0 / ds_real))  # ~1 m circular moving average
        kappa = np.convolve(np.r_[kappa[-win:], kappa, kappa[:win]],
                            np.ones(2 * win + 1) / (2 * win + 1), mode='same')[win:-win]
        v = np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-3))
        v = np.clip(v, v_min, v_max)
        N = len(v)
        for _ in range(2):  # twice around the loop so the limits wrap
            for i in range(1, 2 * N):  # forward: acceleration limit
                j, k = i % N, (i - 1) % N
                v[j] = min(v[j], np.sqrt(v[k] ** 2 + 2 * a_lon * ds_real))
            for i in range(2 * N - 2, -1, -1):  # backward: braking limit
                j, k = i % N, (i + 1) % N
                v[j] = min(v[j], np.sqrt(v[k] ** 2 + 2 * a_lon * ds_real))
        v = np.clip(v, v_min, v_max)
    return np.column_stack([x, y, yaw, v])


class MPPI(Node):
    """
    Implement Kinematic MPPI on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mppi_node')
        # launch/levine_launch.py sets track:=levine, launch/spielberg_launch.py track:=spielberg
        self.declare_parameter('track', 'levine')
        track = self.get_parameter('track').value

        # TODO: create ROS subscribers and publishers
        #       use MPPI as a tracker (similar to pure pursuit)
        # TODO: load the waypoints of `track` from the package's installed waypoints/ folder:
        #       os.path.join(get_package_share_directory('mppi'), 'waypoints', <your csv>)
        #       (ament_index_python.packages). A path like /home/you/... only exists on your laptop.
        # ---- sim or real car: one switch, different defaults (same scheme as the MPC lab) ----
        self.declare_parameter('use_sim', True)  # True = simulator (default, autograder), False = real car
        self.use_sim = bool(self.get_parameter('use_sim').value)
        if self.use_sim:
            default_odom_topic = '/ego_racecar/odom'  # simulator ground-truth odometry (pose + speed)
            default_speed_scale = 1.0                 # full speed profile of the csv
        else:
            default_odom_topic = '/pf/pose/odom'      # particle filter pose on the car
            default_speed_scale = 0.5                 # start at half speed on the car
        self.get_logger().info(f'mode: {"SIM" if self.use_sim else "REAL CAR"}')
        self.declare_parameter('odom_topic', default_odom_topic)
        self.declare_parameter('speed_topic', '/odom')   # real car only: VESC odometry (has the speed)
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('waypoint_file', '')      # '' -> waypoints/<track>.csv
        self.declare_parameter('speed_scale', default_speed_scale)  # scales the speed profile
        self.declare_parameter('max_speed', float(MPPIConfig.MAX_SPEED))  # MPPI speed cap [m/s]
        self.declare_parameter('num_samples', int(MPPIConfig.T))  # T: lower it if the car's CPU is slow
        self.declare_parameter('visualize', False)  # rollouts in RViz: debugging / sim video only
        # latency compensation: predict the state forward by the control delay before planning.
        # -1.0 = automatic (measured time of one control step), 0.0 = off, > 0 = fixed delay [s]
        self.declare_parameter('latency_comp', -1.0)
        # speed command while accelerating: v + a0 * accel_lead_time (README: v + a0 * DT).
        # The simulator's speed controller is proportional (gain ~4.76 when speeding up), so
        # v + a0*DT only gives ~0.7 m/s^2; 0.21 s (= 1/4.76) gives the planned acceleration.
        self.declare_parameter('accel_lead_time', float(MPPIConfig.DT))

        # ---- waypoints ----
        wp_dir = os.path.join(get_package_share_directory('mppi'), 'waypoints')
        opts = WAYPOINTS.get(track, WAYPOINTS['levine'])
        wp_path = os.path.join(wp_dir, self.get_parameter('waypoint_file').value or f'{track}.csv')
        if not os.path.isfile(wp_path):
            self.get_logger().warn(f'{wp_path} not found, using {opts["fallback"]}')
            wp_path = os.path.join(wp_dir, opts['fallback'])
        x_col, y_col, v_col = detect_columns(wp_path)
        self.waypoints = load_waypoints(wp_path, x_col, y_col, v_col, **opts)
        self.waypoints[:, 3] *= float(self.get_parameter('speed_scale').value)
        self.get_logger().info(f"track '{track}': {len(self.waypoints)} waypoints from {wp_path} "
                               f"(speed {'from csv' if v_col is not None else 'from curvature'}), "
                               f"v_ref in [{self.waypoints[:, 3].min():.1f}, {self.waypoints[:, 3].max():.1f}] m/s")

        # ---- topics ----
        qos = QoSProfile(depth=1)  # only the newest pose matters
        self.odom_sub = self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.odom_callback, qos)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.get_parameter('drive_topic').value, 1)
        self.vis_pub = self.create_publisher(MarkerArray, '/mppi/visualization', 1)
        # real car: the particle filter's odometry has the pose but its speed is ~0,
        # so the speed comes from the VESC odometry. In sim /ego_racecar/odom has both
        # (and the autograder only allows /ego_racecar/odom, /tf, /scan, /initialpose).
        self.measured_speed = None
        if not self.use_sim:
            self.speed_sub = self.create_subscription(
                Odometry, self.get_parameter('speed_topic').value, self.speed_callback, qos)

        self.config = MPPIConfig()
        self.U = jnp.zeros((self.config.H, self.config.NU))  # mean control sequence
        self.key = jax.random.PRNGKey(0)

        # jax.jit compiles mppi_step the first time it is called, which takes a
        # few seconds: call it once here so the car does not wait for it
        mppi_step(self.key, self.U, np.zeros(self.config.NX),
                  np.zeros((self.config.H + 1, self.config.NX)), self.config)

        # added: ROS-parameter overrides of the config (new config -> compile once more)
        overrides = dict(MAX_SPEED=float(self.get_parameter('max_speed').value),
                         T=int(self.get_parameter('num_samples').value))
        if any(getattr(self.config, k) != val for k, val in overrides.items()):
            self.config = dataclasses.replace(self.config, **overrides)
            mppi_step(self.key, self.U, np.zeros(self.config.NX),
                      np.zeros((self.config.H + 1, self.config.NX)), self.config)

        # reference speeds above the speed cap would fight the speed penalty: clip them
        # (raise the cap with the max_speed parameter, e.g. max_speed:=8.0 for an 8 m/s csv)
        self.waypoints[:, 3] = np.clip(self.waypoints[:, 3], 0.0, self.config.MAX_SPEED)

        # added: state for the warm start / covariance bonus
        self.Sigma = _prior_sigma(self.config)
        if self.config.ADAPTIVE_COV:
            mppi_step_cov(self.key, self.U, self.Sigma, np.zeros(self.config.NX),
                          np.zeros((self.config.H + 1, self.config.NX)), self.config)
        self.last_t = None    # time [s] of the odom message U was last aligned to
        self.last_xy = None
        self.cold = True
        self.comp_time = 0.0         # EMA of the time one control step takes [s]
        self.last_cmd = (0.0, 0.0)   # last published (acceleration, steering)
        self.last_log = 0.0
        self.last_cb = None   # wall time of the previous control step
        self.max_gap = 0.0    # longest gap between control steps since the last log line

    def odom_callback(self, odom_msg):
        cfg = self.config
        # TODO: extract the state x0 = [p_x, p_y, yaw, v] from the ROS msg
        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
        v = odom_msg.twist.twist.linear.x  # body frame: forward speed
        if not self.use_sim and self.measured_speed is not None:
            v = self.measured_speed  # real car: speed from the VESC
        x0 = np.array([p.x, p.y, yaw, v])
        t_start = time.perf_counter()
        if self.last_cb is not None:
            self.max_gap = max(self.max_gap, t_start - self.last_cb)
        self.last_cb = t_start

        # latency compensation: the command computed now reaches the car only after this step's
        # compute time, so plan from where the car will be then (last command held meanwhile)
        lat = float(self.get_parameter('latency_comp').value)
        if lat < 0.0:
            lat = min(self.comp_time, 0.2)
        if self.last_t is None:  # first message / cold start: nothing to compensate yet
            lat = 0.0
        if lat > 1e-3:
            a_prev, d_prev = self.last_cmd
            n_sub = int(np.ceil(lat / 0.01))
            h = lat / n_sub
            for _ in range(n_sub):
                x0 = x0 + h * np.array([x0[3] * np.cos(x0[2]), x0[3] * np.sin(x0[2]),
                                        x0[3] * np.tan(d_prev) / cfg.WB, a_prev])
            x0[3] = float(np.clip(x0[3], cfg.MIN_SPEED, cfg.MAX_SPEED))

        # TODO: Calculate the reference trajectory for the next H steps
        #       with the current vehicle state.
        #       ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
        wps = self.waypoints
        ref = self.calc_ref_trajectory(x0, wps[:, 0], wps[:, 1], wps[:, 2], wps[:, 3])

        # TODO: warm start: the first input of self.U was applied at the last
        #       step, so shift self.U one step forward (drop the first input
        #       and repeat the last one)
        # Shift by the number of DT steps elapsed since the last shift (odom can arrive
        # faster than 1/DT); reset if time jumps backwards or the car is teleported.
        t = odom_msg.header.stamp.sec + 1e-9 * odom_msg.header.stamp.nanosec
        jumped = self.last_xy is not None and np.hypot(p.x - self.last_xy[0], p.y - self.last_xy[1]) > 1.0
        if self.last_t is None or jumped or t < self.last_t or t - self.last_t > cfg.H * cfg.DT:
            self.U = jnp.zeros((cfg.H, cfg.NU))
            self.Sigma = _prior_sigma(cfg)
            self.last_t = t
            self.cold = True
        else:
            n = int((t - self.last_t) / cfg.DT + 1e-6)
            if n > 0:
                self.U = jnp.concatenate([self.U[n:], jnp.repeat(self.U[-1:], n, axis=0)], axis=0)
                self.Sigma = jnp.concatenate(
                    [self.Sigma[n:], jnp.repeat(_prior_sigma(cfg)[-1:], n, axis=0)], axis=0)
                self.last_t += n * cfg.DT
        self.last_xy = (p.x, p.y)

        # a new random key at every step, so MPPI draws new samples
        self.key, subkey = jax.random.split(self.key)
        # TODO: run MPPI, keep the new mean control sequence in self.U
        # From a cold start (U = 0) a single update is noisy, so iterate a few times
        n_iter = cfg.N_ITER_COLD if self.cold else cfg.N_ITER
        self.cold = False
        for it in range(n_iter):
            if it > 0:
                self.key, subkey = jax.random.split(self.key)
            if cfg.ADAPTIVE_COV:
                self.U, X, w, self.Sigma = mppi_step_cov(subkey, self.U, self.Sigma, x0, ref, cfg)
            else:
                self.U, X, w = mppi_step(subkey, self.U, x0, ref, cfg)

        # TODO: publish drive message: steering delta_0, speed v + a_0 * DT
        #       (float() turns a JAX array into a number)
        a0 = float(np.clip(float(self.U[0, 0]), cfg.MIN_ACCEL, cfg.MAX_ACCEL))
        delta0 = float(np.clip(float(self.U[0, 1]), cfg.MIN_STEER, cfg.MAX_STEER))
        drive = AckermannDriveStamped()
        drive.header.stamp = self.get_clock().now().to_msg()
        drive.header.frame_id = 'ego_racecar/base_link' if self.use_sim else 'base_link'
        drive.drive.steering_angle = delta0
        lead = float(self.get_parameter('accel_lead_time').value) if a0 > 0.0 else cfg.DT
        drive.drive.speed = float(np.clip(v + a0 * lead, cfg.MIN_SPEED, cfg.MAX_SPEED))
        self.drive_pub.publish(drive)
        self.last_cmd = (a0, delta0)

        # timing (EMA) for latency compensation + a log line every 2 s
        dt_step = time.perf_counter() - t_start
        if n_iter == 1:  # cold starts run several iterations: not representative of the delay
            self.comp_time = dt_step if self.comp_time == 0.0 else 0.9 * self.comp_time + 0.1 * dt_step
        if t_start - self.last_log > 2.0:
            self.last_log = t_start
            ess = 1.0 / float(np.sum(np.asarray(w)[:, 0] ** 2))
            msg = (f'control step {1e3 * self.comp_time:.1f} ms, max gap {1e3 * self.max_gap:.0f} ms, '
                   f'v={v:.2f} m/s, ESS={ess:.0f}/{cfg.T}')
            if self.max_gap > 0.1:  # the car drove blind for > 0.1 s: VM / visualization hiccup
                self.get_logger().warn(msg + '  <- hiccup')
            else:
                self.get_logger().info(msg)
            self.max_gap = 0.0

        # TODO (optional, debugging only): visualize the reference, the trajectory
        #       of self.U and a few sampled trajectories, colored by their weight
        if self.get_parameter('visualize').value:
            self.publish_markers(odom_msg.header.frame_id or 'map', x0, ref, X, w)

    # ------------------------------------------------------------------ added
    def speed_callback(self, msg):
        """real car only: forward speed from the VESC odometry"""
        self.measured_speed = msg.twist.twist.linear.x

    def publish_markers(self, frame, x0, ref, X, w, n_show=50):
        cfg = self.config
        now = self.get_clock().now().to_msg()
        X = np.asarray(X)
        w0 = np.asarray(w)[:, 0]
        idx = np.linspace(0, cfg.T - 1, min(n_show, cfg.T)).astype(int)
        w_show = w0[idx] / (w0[idx].max() + 1e-12)

        def line(mid, pts, color, width):
            m = Marker()
            m.header.frame_id, m.header.stamp = frame, now
            m.ns, m.id, m.type, m.action = 'mppi', mid, Marker.LINE_STRIP, Marker.ADD
            m.scale.x = width
            m.pose.orientation.w = 1.0
            m.color = color
            m.points = [Point(x=float(px), y=float(py), z=0.05) for px, py in pts]  # above the map plane
            return m

        arr = MarkerArray()
        for i, (k, s) in enumerate(zip(idx, w_show)):
            # low weight: blue and faint, high weight: red and opaque
            arr.markers.append(line(i, X[k, :, :2], ColorRGBA(r=float(s), g=0.2, b=float(1 - s),
                                                             a=float(0.3 + 0.7 * s)), 0.04))
        arr.markers.append(line(n_show, ref[:, :2], ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), 0.10))
        nominal = np.asarray(_simulate_jit(x0, self.U, cfg))
        arr.markers.append(line(n_show + 1, nominal[:, :2], ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0), 0.10))
        self.vis_pub.publish(arr)

    def calc_ref_trajectory(self, x0, cx, cy, cyaw, cv):
        """
        Calculate interpolated reference trajectory for H steps: [x, y, yaw, v]

        :param x0: current state [p_x, p_y, yaw, v]
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param cv: speed profile
        :return: reference trajectory ref (H+1 x NX)
        """
        return calc_interpolated_ref_trajectory(
            x0[0], x0[1],
            cx, cy, cyaw, cv,
            self.config.DT,
            self.config.H
        )


def main(args=None):
    rclpy.init(args=args)
    print("MPPI Initialized")
    mppi_node = MPPI()
    try:
        rclpy.spin(mppi_node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl-C: stop quietly
    finally:
        mppi_node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()