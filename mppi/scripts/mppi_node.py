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
import sys
import types
try:
    import numba  # noqa: F401
except Exception:  # car: numba missing / broken -> run utils' @njit functions as plain Python
    def _njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f
    sys.modules['numba'] = types.SimpleNamespace(njit=_njit)
from utils import calc_interpolated_ref_trajectory

# TODO CHECK: include needed ROS msg type headers and libraries
import dataclasses
import gc
import os
import time
import re
from collections import deque
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from scipy.ndimage import distance_transform_edt

# added: keep compiled programs on disk, so a restart with the same config loads the warm-up
# compiles instead of compiling again (a config change still compiles once, then is cached too)
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/mppi_jax"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


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
    # adaptive temperature: > 0 picks lambda at every step so the effective sample size
    # ESS = 1 / sum(w^2) is about ESS_TARGET * T. The [0, 1] normalization makes a fixed
    # LAMBDA depend on the worst sample: one collision sample squeezes everything else
    # near 0 (ESS ~T/2, plain average), none spreads them out (ESS ~1, only the best counts).
    # 0 = fixed LAMBDA (the skeleton's behavior)
    ESS_TARGET: float = 0.0
    # obstacle avoidance from /scan (only used when avoid_obstacles is True)
    OBS_RES: float = 0.05      # [m] cell size of the local grid built from the scan
    OBS_CELLS: int = 160       # grid is OBS_CELLS x OBS_CELLS around the car (160 * 0.05 = 8 m)
    # 5 circles of radius 0.17 cover the 0.58 x 0.31 m car (corners are 0.163 m from the end
    # circles) while staying only 1.5 cm wider than the car, so ~0.4 m gaps are still passable
    OBS_CIRCLES: tuple = (-0.24, -0.12, 0.0, 0.12, 0.24)  # [m] circle centers along the car
    OBS_RADIUS: float = 0.17   # [m] circle center closer than this to a scan point = collision
    OBS_MARGIN: float = 0.22   # [m] start the soft penalty at this clearance
    OBS_HARD: float = 100.0    # cost per step in collision
    OBS_SOFT: float = 100.0    # weight on (OBS_MARGIN - d)^2


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


def obstacle_cost(X, obs, cfg):
    """
    Added: per-step obstacle cost of one trajectory, shape (H,)
    :param X: states x_0 ... x_H, shape (H+1, NX)
    :param obs: (dist, ox, oy) local grid from the scan: dist[r, c] = clearance [m] of the cell
                whose lower-left corner is (ox + c*res, oy + r*res)
    """
    dist, ox, oy = obs
    n = dist.shape[0]
    # car footprint = a few circles along the heading (one point would miss the corners)
    off = jnp.array(cfg.OBS_CIRCLES)                                   # (K,)
    px = X[1:, 0:1] + jnp.cos(X[1:, 2:3]) * off                        # (H, K)
    py = X[1:, 1:2] + jnp.sin(X[1:, 2:3]) * off
    c = jnp.floor((px - ox) / cfg.OBS_RES).astype(jnp.int32)
    r = jnp.floor((py - oy) / cfg.OBS_RES).astype(jnp.int32)
    inside = (r >= 0) & (r < n) & (c >= 0) & (c < n)
    clr = dist[jnp.clip(r, 0, n - 1), jnp.clip(c, 0, n - 1)]
    clr = jnp.where(inside, clr, cfg.OBS_MARGIN)  # outside the grid: nothing seen, no cost
    clr = jnp.min(clr, axis=1)                    # the closest circle counts
    return cfg.OBS_HARD * (clr < cfg.OBS_RADIUS) \
        + cfg.OBS_SOFT * jnp.maximum(cfg.OBS_MARGIN - clr, 0.0) ** 2


def _lambda_for_ess(S_n, target, n_iter=16):
    """
    Added: per-step temperature lambda (1, H) with ESS(softmax(-S_n / lambda)) ~= target,
    by bisection on log10(lambda) in [-4, 1] (ESS grows with lambda; with S_n in [0, 1],
    1e-4 keeps only the best sample and 10 weighs all of them about equally)
    """
    def ess(log_lam):
        w = jax.nn.softmax(-S_n / 10.0 ** log_lam, axis=0)
        return 1.0 / jnp.sum(w ** 2, axis=0, keepdims=True)

    def body(_, lo_hi):
        lo, hi = lo_hi
        mid = 0.5 * (lo + hi)
        too_few = ess(mid) < target
        return jnp.where(too_few, mid, lo), jnp.where(too_few, hi, mid)

    shape = (1, S_n.shape[1])
    lo, hi = jax.lax.fori_loop(0, n_iter, body, (jnp.full(shape, -4.0), jnp.full(shape, 1.0)))
    return 10.0 ** (0.5 * (lo + hi))


def _mppi_update(key, U, x0, ref, Sigma, cfg, obs=None):
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    ref = jnp.asarray(ref, dtype=jnp.float32)

    # 1. Sample
    u = _sample(key, U, Sigma, cfg)                                   # (T, H, NU)
    # 2. Simulate
    X = rollout(x0, u, cfg)                                           # (T, H+1, NX)
    # 3. Score: per-step cost, then cost-to-go
    c = jax.vmap(lambda Xk, uk: cost(Xk, uk, ref, cfg))(X, u)         # (T, H)
    if obs is not None:  # added: obstacles seen by the scan
        c = c + jax.vmap(lambda Xk: obstacle_cost(Xk, obs, cfg))(X)
    S =jnp.flip(jnp.cumsum(jnp.flip(c, axis=1), axis=1), axis=1)     # S[k,t] = sum_{t'>=t} c[k,t']
    # 4. Weight: normalize each step to [0, 1] over the samples, softmax over the samples
    S_min = jnp.min(S, axis=0, keepdims=True)
    S_max = jnp.max(S, axis=0, keepdims=True)
    S_n = (S - S_min) / (S_max - S_min + 1e-6)
    lam = _lambda_for_ess(S_n, cfg.ESS_TARGET * cfg.T) if cfg.ESS_TARGET > 0.0 else cfg.LAMBDA
    w = jax.nn.softmax(-S_n / lam, axis=0)                            # (T, H), sums to 1 over T
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


@partial(jax.jit, static_argnames=("cfg",))
def mppi_step_obs(key, U, x0, ref, obs, cfg):
    """Added: mppi_step plus the obstacle cost of the local scan grid obs = (dist, ox, oy)"""
    U_new, X, w, _ = _mppi_update(key, U, x0, ref, _prior_sigma(cfg), cfg, obs)
    return U_new, X, w


@partial(jax.jit, static_argnames=("cfg",))
def mppi_step_cov_obs(key, U, Sigma, x0, ref, obs, cfg):
    """Added: mppi_step_cov plus the obstacle cost"""
    U_new, X, w, Sigma_w = _mppi_update(key, U, x0, ref, Sigma, cfg, obs)
    Sigma_new = (1.0 - cfg.COV_ALPHA) * _prior_sigma(cfg) + cfg.COV_ALPHA * Sigma_w
    Sigma_new = Sigma_new + jnp.diag(jnp.array(cfg.COV_MIN_STD) ** 2)[None]
    return U_new, X, w, Sigma_new


@partial(jax.jit, static_argnames=("cfg",))
def shift_warm_start(U, Sigma, n, cfg):
    """Added: shift U and Sigma n steps forward (repeat the last input, refill Sigma with
    the prior). n is traced, not a slice size, so the shapes never change and a new n
    does not recompile while driving."""
    idx = jnp.arange(cfg.H) + n
    keep = jnp.minimum(idx, cfg.H - 1)
    U_new = U[keep]
    Sigma_new = jnp.where((idx < cfg.H)[:, None, None], Sigma[keep], _prior_sigma(cfg))
    return U_new, Sigma_new


def scan_to_grid(ranges, angles, range_min, px, py, yaw, lidar_offset, cfg, blind_radius=0.8,
                 min_valid=0.1):
    """
    Added: LaserScan -> local clearance grid around the car, axis-aligned in the map frame.
    Only cells the scan has actually seen as empty (between the laser and each return) count
    as free. Everything else -- the returns, the inside and far side of obstacles, the space
    behind walls -- counts as blocked, so a gap that only looks open because its far side was
    never seen is not used. Everything within blind_radius of the car is free (unless hit).
    Returns shorter than min_valid are dropped: the car's laser reports ~0.002 m returns off its
    own body, which would put an "obstacle" inside the car and make every sample collide.
    :return: (dist (OBS_CELLS, OBS_CELLS) float32, ox, oy), same layout as obstacle_cost expects.
        numpy, not jnp: every jnp.asarray / jnp.float32 here is its own GPU op (~2-5 ms each on
        the car), while the jitted MPPI step copies numpy arguments to the GPU with the call
    """
    n, res = cfg.OBS_CELLS, cfg.OBS_RES
    half = 0.5 * n * res
    ox, oy = px - half, py - half
    if len(ranges) == 0:  # no scan (e.g. the warm-up compile): everything free
        return (np.full((n, n), 10.0, dtype=np.float32), np.float32(ox), np.float32(oy))
    lx0 = px + lidar_offset * np.cos(yaw)
    ly0 = py + lidar_offset * np.sin(yaw)
    r_max = 1.45 * half  # far enough to reach the grid corners
    valid = np.isfinite(ranges) & (ranges > max(range_min, min_valid))
    hit = valid & (ranges < r_max)
    r_free = np.where(hit, ranges, r_max)
    r_free = np.where(valid | ~np.isfinite(ranges), r_free, 0.0)  # dropped returns: nothing known

    # cos / sin once per beam, not once per sample point along it (~800 vs ~90k)
    cos_b = np.cos(yaw + angles)
    sin_b = np.sin(yaw + angles)

    def to_cells(r, b):
        wx = lx0 + r * cos_b[b]
        wy = ly0 + r * sin_b[b]
        col = ((wx - ox) / res).astype(np.int32)
        row = ((wy - oy) / res).astype(np.int32)
        keep = (row >= 0) & (row < n) & (col >= 0) & (col < n)
        return row[keep], col[keep]

    free = np.zeros((n, n), dtype=bool)
    # free space: one point per cell length along each beam, up to the return (the return's
    # own cell is set back to blocked below)
    t = np.arange(0.0, r_max, res, dtype=np.float32)                        # (S,)
    along = t[None, :] < (r_free[:, None] - 0.25 * res)                     # (B, S)
    bi, si = np.nonzero(along)
    rr, cc = to_cells(t[si], bi)
    free[rr, cc] = True
    # around the car itself (the car body behind the laser and the laser's rear blind spot are
    # never scanned, but nothing else can be where the car is): free unless the scan hit it
    k = int(np.ceil(blind_radius / res))
    c0 = int((px - ox) / res)
    r0 = int((py - oy) / res)
    ys, xs = np.mgrid[max(r0 - k, 0):min(r0 + k + 1, n), max(c0 - k, 0):min(c0 + k + 1, n)]
    near = np.hypot((xs + 0.5) * res + ox - px, (ys + 0.5) * res + oy - py) < blind_radius
    free[ys[near], xs[near]] = True
    # the returns themselves are blocked
    rr, cc = to_cells(ranges[hit], hit)
    free[rr, cc] = False
    dist = distance_transform_edt(free) * res  # distance of every cell to the nearest blocked one
    return (dist.astype(np.float32), np.float32(ox), np.float32(oy))


def obstacle_speed_factor(ranges, angles, range_min, f_min=0.35,
                          front_half_angle=0.3, front_stop=0.3, front_free=3.0,
                          side_stop=0.2, side_free=0.6):
    """
    Added: reference speed factor in [f_min, 1] from the scan, so the car is already slow
    when it reaches a narrow gap (braking from 3 m/s takes ~1.5 m):
      - front: closest return within +-front_half_angle rad, linear from front_stop to front_free
      - side:  closest return within +-90 deg, linear from side_stop to side_free
    The smaller of the two wins.
    """
    ok = np.isfinite(ranges) & (ranges > max(range_min, 0.1))  # drop the car's own ~0 m returns

    def lin(sel, d0, d1):
        if not sel.any():
            return 1.0
        d = float(np.min(ranges[sel]))
        return float(np.clip(f_min + (1.0 - f_min) * (d - d0) / (d1 - d0), f_min, 1.0))

    f_front = lin(ok & (np.abs(angles) < front_half_angle), front_stop, front_free)
    f_side = lin(ok & (np.abs(angles) < 0.5 * np.pi), side_stop, side_free)
    return min(f_front, f_side)


def plan_lateral_offsets(path, obs, cfg, need, scan_view=None, start_offset=0.0,
                         max_offset=1.0, step=0.05, w_smooth=2.0, lead=0, hold_blind=False):
    """
    Added: bend the reference around obstacles. For the next few meters of the waypoint path
    (path: (M, 4) [x, y, yaw, v], ~0.1 m apart) pick a sideways offset at every point by dynamic
    programming over a grid of offsets, so that every point keeps >= need clearance in the scan
    grid and the offset changes by at most one grid step (step) per point (a smooth curve the car
    can follow). One offset for the whole horizon cannot pass a slalom; a per-point one can.
      cost = sum |offset| + w_smooth * |change| ; blocked cells cost 1e3 (soft, so DP always ends)
    A shifted point must also be in front of the scan return in its direction (not behind a wall),
    unless no offset at that point is visible at all (e.g. around a corner): then the grid decides.
    Minimizing sum |offset| keeps the path on the waypoints as long as possible, so a detour starts
    at the last moment (~0.6 m before the obstacle at one step per point) -- later than the car can
    react (~0.35 s of pose + compute + steering delay). Added for that:
      lead: a point counts as blocked if its offset is blocked within the next `lead` points, so the
            detour is complete that many points before the obstacle
      hold_blind: points where every offset is blocked (typically the unseen space behind an
            obstacle) cost nothing, so the DP keeps the detour it is on there; without it they all
            cost 1e3 and |offset| pulls the path back to 0, i.e. no detour until the car sees around
    lead=0, hold_blind=False is the original planner.
    :return: offsets (M,)
    """
    offs = np.arange(-max_offset, max_offset + 1e-9, step)                   # (K,)
    node = _offset_costs(path, obs, cfg, need, offs, scan_view, lead, hold_blind)
    M, K = node.shape
    k0 = int(np.argmin(np.abs(offs - start_offset)))
    INF = 1e12
    J = np.full(K, INF); J[k0] = node[0, k0]
    back = np.zeros((M, K), dtype=int)
    # max one step per point: offset k comes from k+1, k or k-1 (rows dk = -1, 0, 1 below).
    # One (3, K) array per point instead of a np.roll per move (np.roll was ~70% of the time)
    k_idx = np.arange(K)
    move = (w_smooth * step * np.array([1.0, 0.0, 1.0]))[:, None]          # (3, 1)
    Jp = np.full(K + 2, INF)                                                 # J padded by INF
    for m in range(1, M):
        Jp[1:-1] = J
        cand = np.stack((Jp[2:], Jp[1:-1], Jp[:-2])) + move                 # (3, K)
        d = np.argmin(cand, axis=0)                                          # first min wins
        J = cand[d, k_idx] + node[m]
        back[m] = k_idx - (d - 1)                                            # k - dk
    k = int(np.argmin(J))
    out = np.empty(M)
    for m in range(M - 1, -1, -1):
        out[m] = offs[k]
        k = back[m, k]
    return out


def _offset_costs(path, obs, cfg, need, offs, scan_view=None, lead=0, hold_blind=False):
    """
    Added: cost of every (path point, sideways offset) for the detour planners, shape (M, K):
    |offset| + 1e3 where the offset is blocked (see plan_lateral_offsets for scan_view / lead / hold_blind)
    """
    dist = np.asarray(obs[0]); ox, oy = float(obs[1]), float(obs[2])
    n = dist.shape[0]
    nx, ny = -np.sin(path[:, 2]), np.cos(path[:, 2])                         # left normals
    px = path[:, 0:1] + offs[None, :] * nx[:, None]                          # (M, K)
    py = path[:, 1:2] + offs[None, :] * ny[:, None]
    c = np.floor((px - ox) / cfg.OBS_RES).astype(int)
    r = np.floor((py - oy) / cfg.OBS_RES).astype(int)
    inside = (r >= 0) & (r < n) & (c >= 0) & (c < n)
    clr = np.full(px.shape, np.inf)
    clr[inside] = dist[r[inside], c[inside]]
    free = clr >= need
    if scan_view is not None:
        ranges, angles, lx, ly, yaw = scan_view
        d = np.hypot(px - lx, py - ly)
        b = np.arctan2(py - ly, px - lx) - yaw
        b = np.arctan2(np.sin(b), np.cos(b))
        in_fov = (b >= angles[0]) & (b <= angles[-1])
        idx = np.clip(np.round((b - angles[0]) / (angles[1] - angles[0])).astype(int), 0, len(angles) - 1)
        rr = ranges[idx]
        seen = ~in_fov | ~np.isfinite(rr) | (d < rr - 0.1)
        any_seen = seen.any(axis=1, keepdims=True)
        free &= seen | ~any_seen
    M, K = free.shape
    blocked = ~free
    blind = blocked.all(axis=1) if hold_blind else np.zeros(M, dtype=bool)  # (M,)
    if lead > 0:
        # spread real obstacles (not the blind points) `lead` points back AND forward: MPPI tracks
        # points up to a horizon ahead, so a path that turns back right after the obstacle gets its
        # corner cut into the obstacle (seen in the closed-loop sim)
        src = blocked & ~blind[:, None]
        spread = src.copy()
        for s in range(1, min(lead, M - 1) + 1):
            spread[:-s] |= src[s:]
            spread[s:] |= src[:-s]
        blocked = blocked | spread
    node = np.abs(offs)[None, :] + np.where(blocked, 1e3, 0.0)               # (M, K)
    node[blind] = 0.0
    return node


def plan_lateral_offsets_curv(path, obs, cfg, need, start_offset=0.0, start_slope=0.0,
                              kmax_left=1.0, kmax_right=1.0, max_offset=1.0, station=0.5, step=0.1,
                              max_slope=0.5, w_smooth=2.0, lead=0, hold_blind=False, prev=None, w_prev=0.0):
    """
    Added: plan_lateral_offsets with the car's turning limit. bags wobble_1103_1647 (99 s / 163 s): an
    obstacle on the apex of a hairpin the waypoints already drive at ~R 1.07 m (the car's limit ~1.05 m).
    The point-by-point DP picked the inside gap but has no idea of curvature, so the bent path asked for
    a much tighter turn than the car can do, and MPPI hesitated between straight and full left until it
    was too late. Here the offsets are chosen on stations `station` m apart, and every three stations
    in a row form a curve whose curvature must stay within [-kmax_right, kmax_left] (left > 0). The
    first curve starts at the car: its offset (start_offset) and heading (start_slope = tan(yaw - path
    yaw)), so an inside line that should have been started earlier is seen as undrivable.
      cost = sum |offset| + w_smooth * |change| (+ w_prev * |offset - prev|) per path point,
             + 1e3 for every blocked point or too sharp curve (soft, so the DP always ends)
    prev: the last plan on the current path points (nan where none): w_prev > 0 keeps the side chosen
    in the last control steps instead of flipping between two detours from one scan to the next.
    :return: offsets (M,), True if no point is blocked and every curve is drivable
    """
    M = len(path)
    ds = float(np.mean(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1])))) if M > 1 else 0.1
    fstep = 0.05
    offs_f = np.arange(-max_offset, max_offset + 1e-9, fstep)               # fine grid of the costs
    node = _offset_costs(path, obs, cfg, need, offs_f, None, lead, hold_blind)
    if prev is not None and w_prev > 0.0:
        has = np.isfinite(prev)
        node[has] += w_prev * np.abs(offs_f[None, :] - prev[has, None])
    L = max(1, int(round(station / ds)))
    sta = np.arange(0, M, L)
    if len(sta) < 3:
        return np.full(M, float(np.clip(start_offset, -max_offset, max_offset))), True
    offs = np.arange(-max_offset, max_offset + 1e-9, step)                   # (K,) station offsets
    K = len(offs)
    nx, ny = -np.sin(path[:, 2]), np.cos(path[:, 2])
    SX = path[sta, 0:1] + offs[None, :] * nx[sta, None]                       # (S, K) station points
    SY = path[sta, 1:2] + offs[None, :] * ny[sta, None]
    INF = 1e12

    def edge(j):
        """(K, K) cost of going from offset k at station j to k' at station j+1"""
        m0, m1 = sta[j], sta[j + 1]
        do = offs[None, :] - offs[:, None]
        c = w_smooth * np.abs(do) + np.where(np.abs(do) > max_slope * (m1 - m0) * ds + 1e-9, INF, 0.0)
        for m in range(m0 + 1, m1 + 1):
            o = offs[:, None] + do * (m - m0) / (m1 - m0)
            kf = np.clip(np.round((o + max_offset) / fstep).astype(int), 0, len(offs_f) - 1)
            c = c + node[m, kf]
        return c

    def kappa(ax, ay, bx, by, cx, cy):
        """signed curvature of the circle through a, b, c (left > 0)"""
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        den = np.hypot(bx - ax, by - ay) * np.hypot(cx - bx, cy - by) * np.hypot(cx - ax, cy - ay) + 1e-9
        return 2.0 * cross / den

    # the waypoints themselves are driven every lap: a station may always be as sharp as the waypoints
    # there (their hairpin is ~R 1.07 m, right at the car's limit, and the 3-point estimate scatters)
    i_mid = int(np.argmin(np.abs(offs)))
    k_base = np.zeros(len(sta))
    k_base[1:-1] = kappa(SX[:-2, i_mid], SY[:-2, i_mid], SX[1:-1, i_mid], SY[1:-1, i_mid], SX[2:, i_mid], SY[2:, i_mid])
    k_base[0] = k_base[1]
    lim_l = np.maximum(kmax_left, 1.05 * k_base)
    lim_r = np.maximum(kmax_right, -1.05 * k_base)

    def curv_cost(j, ax, ay, bx, by, cx, cy):
        """1e3 (+ the relative excess) where the curve at station j is sharper than the car can turn"""
        kap = kappa(ax, ay, bx, by, cx, cy)
        over = np.maximum(kap - lim_l[j], 0.0) / lim_l[j] + np.maximum(-kap - lim_r[j], 0.0) / lim_r[j]
        return np.where(over > 0.0, 1e3 * (1.0 + over), 0.0)

    # station 0 is the car; the point one station behind it continues the car's heading
    k0 = int(np.argmin(np.abs(offs - start_offset)))
    tx, ty = np.cos(path[0, 2]), np.sin(path[0, 2])
    o_back = offs[k0] - start_slope * L * ds
    bx0 = path[0, 0] - L * ds * tx + o_back * nx[0]
    by0 = path[0, 1] - L * ds * ty + o_back * ny[0]
    J = np.full((K, K), INF)                                                   # J[k_prev, k_cur]
    J[k0, :] = edge(0)[k0] + curv_cost(0, bx0, by0, SX[0, k0], SY[0, k0], SX[1], SY[1])
    backs = []
    for j in range(1, len(sta) - 1):
        cc = curv_cost(j, SX[j - 1][:, None, None], SY[j - 1][:, None, None], SX[j][None, :, None],
                       SY[j][None, :, None], SX[j + 1][None, None, :], SY[j + 1][None, None, :])
        tot = J[:, :, None] + cc                                               # (kp, k, kn)
        bp = np.argmin(tot, axis=0)                                            # (k, kn)
        J = np.take_along_axis(tot, bp[None], axis=0)[0] + edge(j)
        backs.append(bp)
    kp, kc = np.unravel_index(int(np.argmin(J)), J.shape)
    best = float(J[kp, kc])
    ks = [kc, kp]
    for bp in reversed(backs):
        ks.append(int(bp[ks[-1], ks[-2]]))
    ks = ks[::-1]                                                              # one per station
    out = np.interp(np.arange(M), sta[:len(ks)], offs[ks])
    return out, best < 1e3


class MapObstacles:
    """
    Added: what the obstacle grid knows besides the newest scan.
    scan_to_grid only trusts space the scan has seen empty, so the gap behind/inside an obstacle on the
    apex of a hairpin stays blocked until the car is through the turn (bag wobble_1103_1647: the inside
    line looked impossible from the entry straight, then possible 1.7 m before the obstacle -- too late).
    Here the static map (the particle filter's /map) says where the walls are, so unseen cells are free
    unless the map has a wall there, and a short memory keeps the obstacles that are not in the map:
      - every scan return not on a map wall counts as a hit; hits fade by `decay` per scan and a cell is
        an obstacle while it has >= hit_thresh (one stray return is not enough, a stationary box is)
      - cells a beam has passed through are cleared (moved obstacles, floor returns seen from closer)
      - a return is assumed to be the front of a box `depth` m deep along the beam (its back is unseen)
    Map layout as nav_msgs/OccupancyGrid: cell (r, c) covers x >= ox + c*res, y >= oy + r*res.
    """
    def __init__(self, wall, res, ox, oy, wall_margin=0.15, decay=0.8, hit_thresh=1.5, mem_range=3.0,
                 depth=0.3):
        self.wall = np.asarray(wall, dtype=bool)             # wall or unknown
        self.res, self.ox, self.oy = float(res), float(ox), float(oy)
        self.wall_dist = (distance_transform_edt(~self.wall) * self.res).astype(np.float32)
        self.mem = np.zeros(self.wall.shape, dtype=np.float32)
        self.wall_margin, self.decay, self.hit_thresh = wall_margin, decay, hit_thresh
        self.mem_range, self.depth = mem_range, depth

    @classmethod
    def from_occupancy_grid(cls, msg, **kw):
        info = msg.info
        data = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        return cls((data >= 50) | (data < 0), info.resolution, info.origin.position.x,
                   info.origin.position.y, **kw)

    def _cells(self, x, y):
        c = np.floor((x - self.ox) / self.res).astype(np.int32)
        r = np.floor((y - self.oy) / self.res).astype(np.int32)
        ok = (r >= 0) & (r < self.wall.shape[0]) & (c >= 0) & (c < self.wall.shape[1])
        return r, c, ok

    def update(self, ranges, angles, range_min, px, py, yaw, lidar_offset):
        """one scan, taken at pose (px, py, yaw)"""
        lx, ly = px + lidar_offset * np.cos(yaw), py + lidar_offset * np.sin(yaw)
        valid = np.isfinite(ranges) & (ranges > max(range_min, 0.1))
        self.mem *= self.decay
        # clear what the beams passed through (every 2nd beam, 0.1 m steps, stop 0.15 m before the return)
        b = np.arange(0, len(ranges), 2)
        reach = np.where(valid[b], np.minimum(ranges[b], self.mem_range + 0.5), 0.0)
        t = np.arange(0.3, self.mem_range + 0.5, 0.1)
        bi, si = np.nonzero(t[None, :] < reach[:, None] - 0.15)
        r, c, ok = self._cells(lx + t[si] * np.cos(yaw + angles[b[bi]]), ly + t[si] * np.sin(yaw + angles[b[bi]]))
        self.mem[r[ok], c[ok]] = 0.0
        # hits: returns within mem_range that are not part of a map wall
        h = valid & (ranges < self.mem_range)
        r, c, ok = self._cells(lx + ranges[h] * np.cos(yaw + angles[h]), ly + ranges[h] * np.sin(yaw + angles[h]))
        r, c = r[ok], c[ok]
        keep = self.wall_dist[r, c] > self.wall_margin
        cells = np.unique(np.c_[r[keep], c[keep]], axis=0)
        self.mem[cells[:, 0], cells[:, 1]] += 1.0

    def local_grid(self, ranges, angles, range_min, px, py, yaw, lidar_offset, cfg):
        """same output as scan_to_grid: (dist, ox, oy) around the car, axis-aligned"""
        n, res = cfg.OBS_CELLS, cfg.OBS_RES
        half = 0.5 * n * res
        gx, gy = px - half, py - half
        ctr = (np.arange(n) + 0.5) * res
        r = np.floor((gy + ctr - self.oy) / self.res).astype(np.int32)        # map row of each grid row
        c = np.floor((gx + ctr - self.ox) / self.res).astype(np.int32)        # map column of each grid column
        ok_r = (r >= 0) & (r < self.wall.shape[0])
        ok_c = (c >= 0) & (c < self.wall.shape[1])
        rows = np.clip(r, 0, self.wall.shape[0] - 1)[:, None]
        cols = np.clip(c, 0, self.wall.shape[1] - 1)[None, :]
        blocked = self.wall[rows, cols] | (self.mem[rows, cols] >= self.hit_thresh)
        blocked |= ~(ok_r[:, None] & ok_c[None, :])                           # off the map
        if len(ranges):
            lx, ly = px + lidar_offset * np.cos(yaw), py + lidar_offset * np.sin(yaw)
            hit = np.isfinite(ranges) & (ranges > max(range_min, 0.1)) & (ranges < 1.45 * half)
            rh, ah = ranges[hit], angles[hit]
            wx, wy = lx + rh * np.cos(yaw + ah), ly + rh * np.sin(yaw + ah)
            mr, mc, mok = self._cells(wx, wy)
            on_wall = np.zeros(len(rh), dtype=bool)
            on_wall[mok] = self.wall_dist[mr[mok], mc[mok]] <= self.wall_margin
            # every return blocks its cell; returns off the map walls also the `depth` behind them
            t = np.arange(0.0, self.depth + 1e-9, res)
            ext = np.where(on_wall[:, None], 0.0, t[None, :])                 # (B, S)
            ex = lx + (rh[:, None] + ext) * np.cos(yaw + ah)[:, None]
            ey = ly + (rh[:, None] + ext) * np.sin(yaw + ah)[:, None]
            gc = np.floor((ex - gx) / res).astype(np.int32)
            gr = np.floor((ey - gy) / res).astype(np.int32)
            k = (gr >= 0) & (gr < n) & (gc >= 0) & (gc < n)
            blocked[gr[k], gc[k]] = True
        dist = distance_transform_edt(~blocked) * res
        return (dist.astype(np.float32), np.float32(gx), np.float32(gy))


def detour_offsets(wps, px, py, yaw, v, obs, scan, lidar_offset, cfg, opts, prev_bend=None):
    """
    Added: the node's bend_ref without the node (the closed-loop test calls this too). Plans sideways
    offsets along the next opts['lookahead'] m of the waypoints wps, starting at the car's own offset.
    :param scan: (ranges, angles) of the newest scan, for the "in front of the return" check
    :param prev_bend: (first path index, offsets) of the last plan, for opts['consistency']
    :return: path (M, 4), offsets (M,), index of path[0] in wps, True if the plan is free and drivable
    """
    N = len(wps)
    i0 = int(np.argmin(np.hypot(wps[:, 0] - px, wps[:, 1] - py)))
    ds = float(np.mean(np.hypot(np.diff(wps[:50, 0]), np.diff(wps[:50, 1]))))
    M = max(int(opts['lookahead'] / ds), 5)
    path = wps[(i0 + np.arange(M)) % N]
    # the car's own sideways offset from the path, so the bent path starts at the car
    nx, ny = -np.sin(path[0, 2]), np.cos(path[0, 2])
    start = (px - path[0, 0]) * nx + (py - path[0, 1]) * ny
    need = cfg.OBS_RADIUS + opts['extra']
    # finish the detour this far before the obstacle: the distance driven during the delay
    lead = int(round((max(v, 0.0) * opts['lead_time'] + opts['lead_dist']) / ds))
    if not opts.get('curvature', False):
        lx = px + lidar_offset * np.cos(yaw)
        ly = py + lidar_offset * np.sin(yaw)
        # with the map prior the grid already knows what is behind the walls: no visibility check
        view = None if opts.get('map_prior', False) else (scan[0], scan[1], lx, ly, yaw)
        o_path = plan_lateral_offsets(path, obs, cfg, need=need, scan_view=view, start_offset=start,
                                      lead=lead, hold_blind=opts['hold_blind'])
        return path, o_path, i0, True
    prev = None
    if prev_bend is not None and opts.get('consistency', 0.0) > 0.0:
        i_prev, o_prev = prev_bend
        shift = (i0 - i_prev) % N
        prev = np.full(M, np.nan)
        if shift < len(o_prev):
            k = min(M, len(o_prev) - shift)
            prev[:k] = o_prev[shift:shift + k]
    dyaw = np.arctan2(np.sin(yaw - path[0, 2]), np.cos(yaw - path[0, 2]))
    scale = opts.get('curv_scale', 1.0)
    o_path, ok = plan_lateral_offsets_curv(
        path, obs, cfg, need, start_offset=start, start_slope=float(np.tan(np.clip(dyaw, -1.2, 1.2))),
        kmax_left=scale * np.tan(cfg.MAX_STEER) / cfg.WB, kmax_right=scale * np.tan(-cfg.MIN_STEER) / cfg.WB,
        station=opts.get('station', 0.5), lead=lead, hold_blind=opts['hold_blind'], prev=prev,
        w_prev=opts.get('consistency', 0.0))
    return path, o_path, i0, ok


def bend_reference(ref, path, o_path):
    """Added: move each reference point sideways by the offset at its distance along path"""
    ds = float(np.mean(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))))
    # distance of every reference point along the path, measured from path[0]
    tx, ty = np.cos(path[0, 2]), np.sin(path[0, 2])
    s0 = (ref[0, 0] - path[0, 0]) * tx + (ref[0, 1] - path[0, 1]) * ty
    s_ref = s0 + np.r_[0.0, np.cumsum(np.hypot(np.diff(ref[:, 0]), np.diff(ref[:, 1])))]
    o_ref = np.interp(s_ref, np.arange(len(path)) * ds, o_path)
    out = np.array(ref, dtype=float)
    out[:, 0] += o_ref * -np.sin(ref[:, 2])
    out[:, 1] += o_ref * np.cos(ref[:, 2])
    return out


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
        # steering limit [rad], +- (the real car's servo range can be smaller than the sim's 0.4189)
        self.declare_parameter('max_steer', float(MPPIConfig.MAX_STEER))
        # added: per-side limits [rad, > 0] for a servo whose range is not centered on straight
        # (steering > 0 = left). < 0 = use max_steer for that side
        self.declare_parameter('max_steer_left', -1.0)
        self.declare_parameter('max_steer_right', -1.0)
        self.declare_parameter('num_samples', int(MPPIConfig.T))  # T: lower it if the car's CPU is slow
        self.declare_parameter('visualize', False)  # rollouts in RViz: debugging / sim video only
        # latency compensation: predict the state forward by the control delay before planning.
        # -1.0 = automatic (measured time of one control step), 0.0 = off, > 0 = fixed delay [s]
        self.declare_parameter('latency_comp', -1.0)
        # True: replay the commands sent during the last latency_comp s; False: hold the last one
        self.declare_parameter('latency_history', False)
        # speed command while accelerating: v + a0 * accel_lead_time (README: v + a0 * DT).
        # The simulator's speed controller is proportional (gain ~4.76 when speeding up), so
        # v + a0*DT only gives ~0.7 m/s^2; 0.21 s (= 1/4.76) gives the planned acceleration.
        self.declare_parameter('accel_lead_time', float(MPPIConfig.DT))
        # added: MPPI weights / sampling as ROS parameters (defaults = MPPIConfig)
        self.declare_parameter('Q', list(MPPIConfig.Q))
        self.declare_parameter('Qf', list(MPPIConfig.Qf))
        self.declare_parameter('R', list(MPPIConfig.R))
        self.declare_parameter('Rd', list(MPPIConfig.Rd))
        self.declare_parameter('lambda', float(MPPIConfig.LAMBDA))
        self.declare_parameter('ess_target', float(MPPIConfig.ESS_TARGET))  # > 0: adaptive lambda
        self.declare_parameter('u_std', list(MPPIConfig.U_STD))
        # added: waypoint direction. 'auto' = the track table (levine -> counter-clockwise, the
        # graded direction), 'keep' = as recorded in the csv, 'ccw' / 'cw' = force
        self.declare_parameter('waypoint_direction', 'auto')

        # ---- waypoints ----
        wp_dir = os.path.join(get_package_share_directory('mppi'), 'waypoints')
        opts = dict(WAYPOINTS.get(track, WAYPOINTS['levine']))
        direction = str(self.get_parameter('waypoint_direction').value).lower()
        if direction in ('keep', 'ccw', 'cw'):
            opts['ccw'] = {'keep': None, 'ccw': True, 'cw': False}[direction]
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
        # random keys made once up front, as numpy: a jax.random.split every step is its own
        # GPU op (~5 ms on the car). 2^16 keys last ~1 h at 20 Hz, then they repeat
        self.keys = np.asarray(jax.random.split(jax.random.PRNGKey(0), 1 << 16))
        self.key_i = 0
        self.key = self.keys[0]  # for the warm-up compiles

        # jax.jit compiles mppi_step for every distinct config (it is a static argument), which
        # takes seconds: the warm-up compile runs once at the end of __init__, with the final
        # config (defaults + the ROS-parameter overrides below + obstacle avoidance)

        # added: ROS-parameter overrides of the config
        max_steer = abs(float(self.get_parameter('max_steer').value))
        steer_left = float(self.get_parameter('max_steer_left').value)
        steer_right = float(self.get_parameter('max_steer_right').value)
        steer_left = steer_left if steer_left >= 0.0 else max_steer
        steer_right = steer_right if steer_right >= 0.0 else max_steer
        overrides = dict(MAX_SPEED=float(self.get_parameter('max_speed').value),
                         MAX_STEER=steer_left,
                         MIN_STEER=-steer_right,
                         T=int(self.get_parameter('num_samples').value),
                         Q=tuple(float(q) for q in self.get_parameter('Q').value),
                         Qf=tuple(float(q) for q in self.get_parameter('Qf').value),
                         R=tuple(float(q) for q in self.get_parameter('R').value),
                         Rd=tuple(float(q) for q in self.get_parameter('Rd').value),
                         LAMBDA=float(self.get_parameter('lambda').value),
                         ESS_TARGET=float(self.get_parameter('ess_target').value),
                         U_STD=tuple(float(q) for q in self.get_parameter('u_std').value))
        if any(getattr(self.config, k) != val for k, val in overrides.items()):
            self.config = dataclasses.replace(self.config, **overrides)
        self.get_logger().info(f'steering limits: left {self.config.MAX_STEER:.3f}, '
                               f'right {-self.config.MIN_STEER:.3f} rad')

        # reference speeds above the speed cap would fight the speed penalty: clip them
        # (raise the cap with the max_speed parameter, e.g. max_speed:=8.0 for an 8 m/s csv)
        self.waypoints[:, 3] = np.clip(self.waypoints[:, 3], 0.0, self.config.MAX_SPEED)

        # added: state for the warm start / covariance bonus
        self.Sigma = _prior_sigma(self.config)
        self.last_t = None    # time [s] of the odom message U was last aligned to
        self.last_xy = None
        self.cold = True
        self.comp_time = 0.0         # EMA of the time one control step takes [s]
        self.last_cmd = (0.0, 0.0)   # last published (acceleration, steering)
        self.cmd_hist = deque(maxlen=64)  # (time.monotonic(), acceleration, steering) per command
        self.last_log = 0.0
        self.last_cb = None   # wall time of the previous control step
        self.max_gap = 0.0    # longest gap between control steps since the last log line

        # ---- added: obstacle avoidance from /scan ----
        self.declare_parameter('avoid_obstacles', False)  # False = original behavior (autograder)
        self.declare_parameter('scan_topic', '/scan')
        # laser position ahead of the odom point [m]: sim = scan_distance_to_base_link in sim.yaml
        self.declare_parameter('lidar_offset', 0.0 if self.use_sim else 0.27)
        self.declare_parameter('steer_std', 0.2)  # wider steering samples so detours get sampled
        # clearance [m] of each footprint circle to the nearest scan point: < obs_radius = collision,
        # < obs_margin = soft penalty. 0.17 covers the car (half width 0.155 m)
        self.declare_parameter('obs_radius', 0.17)
        self.declare_parameter('obs_margin', 0.22)
        # reference speed x factor in [slow_min_factor, 1] from the closest scan return ahead
        self.declare_parameter('slow_near_obstacles', True)
        self.declare_parameter('slow_min_factor', 0.2)
        # half-angle [rad] of the "front" cone for slowing down; wider also catches obstacles
        # the car turns into (in the race2 bag both contacts came in at +40..+60 deg)
        self.declare_parameter('slow_front_angle', 0.3)
        # [m] the factor goes linearly from slow_min_factor at *_stop to 1 at *_free:
        # front = nearest return in the front cone, side = nearest return within +-90 deg
        self.declare_parameter('slow_front_stop', 0.3)
        self.declare_parameter('slow_front_free', 3.0)
        self.declare_parameter('slow_side_stop', 0.2)
        self.declare_parameter('slow_side_free', 0.6)
        # shift the reference sideways when it runs through an obstacle (avoids getting stuck)
        self.declare_parameter('shift_ref', True)
        self.declare_parameter('bend_lookahead', 4.0)  # [m] of path the offsets are planned over
        self.declare_parameter('bend_extra', 0.03)     # [m] bent path keeps obs_radius + this clearance
        # detour complete v * bend_lead_time + bend_lead_dist [m] before the obstacle (0, 0 = original)
        self.declare_parameter('bend_lead_time', 0.0)  # [s]
        self.declare_parameter('bend_lead_dist', 0.0)  # [m]
        # keep the current detour where no offset is free (unseen space behind an obstacle)
        self.declare_parameter('bend_hold_blind', False)
        # added: detour planner that knows the car's turning limit (plan_lateral_offsets_curv); False = original
        self.declare_parameter('bend_curvature', False)
        self.declare_parameter('bend_curv_scale', 1.0)     # x tan(steering limit) / wheelbase per side
        self.declare_parameter('bend_station', 0.5)        # [m] between the planned offsets
        self.declare_parameter('bend_consistency', 0.0)    # weight on |offset - last plan|: no side flipping
        # reference speed x slow_min_factor while the detour planner finds no free, drivable line
        self.declare_parameter('bend_fail_slow', False)
        # added: static map + obstacle memory instead of "unseen = blocked" (MapObstacles)
        self.declare_parameter('use_map_prior', False)
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('obstacle_depth', 0.3)      # [m] assumed depth behind each obstacle return
        self.declare_parameter('obstacle_decay', 0.8)      # memory hits x this per scan
        self.declare_parameter('obstacle_range', 3.0)      # [m] only returns closer than this are remembered
        self.bend_ok = True
        self.prev_bend = None    # (index of the first path point, offsets) of the last detour plan
        self.map_obs = None
        self.scan_seq = 0        # counts scans, so one scan is added to the memory only once
        self.mem_seq = -1
        # stuck recovery: back up when standing still for stuck_time s
        # cold start (after a reset) runs this many MPPI iterations at most. Each iteration costs
        # a full control step, so on a slow computer 10 iterations = a long blind gap, and a gap
        # longer than H*DT triggers the next reset -> keep it small on the car
        self.declare_parameter('cold_iters', int(MPPIConfig.N_ITER_COLD))
        self.t_grid = 0.0
        self.t_bend = 0.0
        self.declare_parameter('recovery', False)   # experimental: unreliable in tests, off
        self.declare_parameter('stuck_time', 1.0)       # [s]
        self.declare_parameter('reverse_time', 0.7)     # [s]
        self.declare_parameter('reverse_speed', -0.6)   # [m/s]
        self.stuck_since = None
        self.recover_until = None
        self.recover_steer = 0.0
        self.avoid = bool(self.get_parameter('avoid_obstacles').value)
        self.lidar_offset = float(self.get_parameter('lidar_offset').value)
        self.scan = None  # (ranges, angles, range_min) of the newest scan
        if self.avoid:
            self.scan_sub = self.create_subscription(
                LaserScan, self.get_parameter('scan_topic').value, self.scan_callback, qos)
            # the scan already sees the walls; the off-track penalty would only fight the detours
            self.config = dataclasses.replace(
                self.config, OFFTRACK_PENALTY=0.0,
                U_STD=(self.config.U_STD[0], float(self.get_parameter('steer_std').value)),
                OBS_RADIUS=float(self.get_parameter('obs_radius').value),
                OBS_MARGIN=float(self.get_parameter('obs_margin').value))
            cfg = self.config
            self.Sigma = _prior_sigma(cfg)
            # before the first scan: an all-free grid (zero obstacle cost), so the control loop
            # always runs the obstacle version of the step and only that one is compiled
            self.empty_grid = scan_to_grid(np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0, cfg)
            self.get_logger().info(f'obstacle avoidance ON: {self.get_parameter("scan_topic").value}, '
                                   f'lidar_offset {self.lidar_offset} m, steer_std {cfg.U_STD[1]}, '
                                   f'obs_radius {cfg.OBS_RADIUS}, obs_margin {cfg.OBS_MARGIN}')
            if self.get_parameter('use_map_prior').value:
                # map_server publishes /map once, latched
                map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                     reliability=ReliabilityPolicy.RELIABLE)
                self.map_sub = self.create_subscription(
                    OccupancyGrid, self.get_parameter('map_topic').value, self.map_callback, map_qos)

        # ---- diagnostics (logging only): where do the ~200 ms on the car come from ----
        # per-segment callback times [s], printed as p50/max with the 2 s log line
        self.seg = {k: [] for k in ('pre', 'ref', 'prep', 'obs', 'mppi', 'pub', 'age')}

        # garbage collector pauses: every gen-2 collection, other generations only if > 5 ms
        self._gc_t0 = None

        def _gc_cb(phase, info):
            if phase == 'start':
                self._gc_t0 = time.perf_counter()
            elif self._gc_t0 is not None:
                dt = time.perf_counter() - self._gc_t0
                if info['generation'] == 2 or dt > 0.005:
                    self.get_logger().warn(
                        f'GC gen-{info["generation"]} took {1e3 * dt:.0f} ms '
                        f'(collected {info["collected"]})')

        gc.callbacks.append(_gc_cb)

        # the warm-up compiles, once, with the final config: exactly the steps the control loop runs
        cfg = self.config
        z_x, z_ref = np.zeros(cfg.NX), np.zeros((cfg.H + 1, cfg.NX))
        t_compile = time.perf_counter()
        if self.avoid and cfg.ADAPTIVE_COV:
            mppi_step_cov_obs(self.key, self.U, self.Sigma, z_x, z_ref, self.empty_grid, cfg)
        elif self.avoid:
            mppi_step_obs(self.key, self.U, z_x, z_ref, self.empty_grid, cfg)
        elif cfg.ADAPTIVE_COV:
            mppi_step_cov(self.key, self.U, self.Sigma, z_x, z_ref, cfg)
        else:
            mppi_step(self.key, self.U, z_x, z_ref, cfg)
        shift_warm_start(self.U, self.Sigma, 1, cfg)
        self.get_logger().info(f'warm-up compile: {time.perf_counter() - t_compile:.1f} s')

        # turned on only after all warm-up compiles: from here on every "Compiling ..."
        # line is a compile while driving
        jax.config.update("jax_log_compiles", True)

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
            # added (latency_history): over the delay the car still gets the commands sent during the
            # last `lat` s, in order -- not the last one held. With lat ~0.3 s holding the last one
            # predicted heading errors of 30+ deg whenever the steering changed (closed-loop sim)
            hist = list(self.cmd_hist) if self.get_parameter('latency_history').value else []
            now = time.monotonic()
            for j in range(n_sub):
                if hist:
                    tau = now - lat + j * h   # the command active then: the newest one sent by tau
                    a_prev, d_prev = 0.0, 0.0
                    for t_cmd, a_c, d_c in hist:
                        if t_cmd > tau:
                            break
                        a_prev, d_prev = a_c, d_c
                x0 = x0 + h * np.array([x0[3] * np.cos(x0[2]), x0[3] * np.sin(x0[2]),
                                        x0[3] * np.tan(d_prev) / cfg.WB, a_prev])
            x0[3] = float(np.clip(x0[3], cfg.MIN_SPEED, cfg.MAX_SPEED))

        # TODO: Calculate the reference trajectory for the next H steps
        #       with the current vehicle state.
        #       ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
        # added: the obstacle slowdown scales the speed profile BEFORE the reference is built, so
        # the reference points are spaced for the slower speed too. Scaling only ref[:, 3] afterwards
        # left the points spaced for full speed, and the position cost (Q x, y = 5) beat the speed
        # cost (Q v = 1): bags wobble_1102_1350 / 1431 drove ~1.9 m/s at any obstacle distance
        scan = self.scan if self.avoid else None
        f_slow = 1.0
        if scan is not None and self.get_parameter('slow_near_obstacles').value:
            f_slow = obstacle_speed_factor(scan[0], scan[1], scan[2],
                                           f_min=float(self.get_parameter('slow_min_factor').value),
                                           front_half_angle=float(self.get_parameter('slow_front_angle').value),
                                           front_stop=float(self.get_parameter('slow_front_stop').value),
                                           front_free=float(self.get_parameter('slow_front_free').value),
                                           side_stop=float(self.get_parameter('slow_side_stop').value),
                                           side_free=float(self.get_parameter('slow_side_free').value))
        if scan is not None and not self.bend_ok and self.get_parameter('bend_fail_slow').value:
            # added: the last detour plan found no free, drivable line -> slow down early
            f_slow = min(f_slow, float(self.get_parameter('slow_min_factor').value))
        wps = self.waypoints
        t_a = time.perf_counter()
        ref = self.calc_ref_trajectory(x0, wps[:, 0], wps[:, 1], wps[:, 2], wps[:, 3] * f_slow)
        t_b = time.perf_counter()

        # TODO: warm start: the first input of self.U was applied at the last
        #       step, so shift self.U one step forward (drop the first input
        #       and repeat the last one)
        # Shift by the number of DT steps elapsed since the last shift (odom can arrive
        # faster than 1/DT); reset if time jumps backwards or the car is teleported.
        t = odom_msg.header.stamp.sec + 1e-9 * odom_msg.header.stamp.nanosec
        # same reset conditions as before, but log which one fired
        reasons = []
        if self.last_t is not None:
            d_jump = np.hypot(p.x - self.last_xy[0], p.y - self.last_xy[1])
            if d_jump > 1.0:
                reasons.append(f'pose jump {d_jump:.2f} m')
            if t < self.last_t:
                reasons.append(f'stamp backwards {1e3 * (t - self.last_t):.0f} ms')
            elif t - self.last_t > cfg.H * cfg.DT:
                reasons.append(f'gap {t - self.last_t:.2f} s')
        if self.last_t is None or reasons:
            if reasons:  # the cold start on the first message is expected: not reported
                self.get_logger().warn('cold reset: ' + ', '.join(reasons))
            self.U = jnp.zeros((cfg.H, cfg.NU))
            self.Sigma = _prior_sigma(cfg)
            self.last_t = t
            self.cold = True
        else:
            n = int((t - self.last_t) / cfg.DT + 1e-6)
            if n > 0:
                self.U, self.Sigma = shift_warm_start(self.U, self.Sigma, n, cfg)
                self.last_t += n * cfg.DT
        self.last_xy = (p.x, p.y)
        t_c = time.perf_counter()

        # added: local clearance grid from the newest scan, placed with the measured pose
        # (not the latency-compensated x0: the scan was taken where the car is now)
        obs = self.empty_grid if self.avoid else None  # no scan yet: nothing seen, no obstacle cost
        if scan is not None:
            ranges, angles, range_min = scan
            tg = time.perf_counter()
            if self.map_obs is not None:  # added: map + obstacle memory (use_map_prior)
                if self.mem_seq != self.scan_seq:
                    self.map_obs.update(ranges, angles, range_min, p.x, p.y, yaw, self.lidar_offset)
                    self.mem_seq = self.scan_seq
                obs = self.map_obs.local_grid(ranges, angles, range_min, p.x, p.y, yaw, self.lidar_offset, cfg)
            else:
                obs = scan_to_grid(ranges, angles, range_min, p.x, p.y, yaw, self.lidar_offset, cfg)
            tb = time.perf_counter()
            if self.get_parameter('shift_ref').value:
                ref = self.bend_ref(np.asarray(ref, dtype=float), p.x, p.y, yaw, obs,
                                    (ranges, angles), cfg, v=v)
            te = time.perf_counter()
            self.t_grid = 0.9 * self.t_grid + 0.1 * (tb - tg)   # timing for the log line
            self.t_bend = 0.9 * self.t_bend + 0.1 * (te - tb)

        # a new random key at every step, so MPPI draws new samples
        subkey = self.next_key()
        t_o = time.perf_counter()
        # TODO: run MPPI, keep the new mean control sequence in self.U
        # From a cold start (U = 0) a single update is noisy, so iterate a few times
        n_iter = cfg.N_ITER_COLD if self.cold else cfg.N_ITER
        if self.cold:  # added: fewer cold-start iterations on a slow computer (car)
            n_iter = min(n_iter, int(self.get_parameter('cold_iters').value))
        self.cold = False
        for it in range(n_iter):
            if it > 0:
                subkey = self.next_key()
            if obs is not None:  # added: same update plus the obstacle cost
                if cfg.ADAPTIVE_COV:
                    self.U, X, w, self.Sigma = mppi_step_cov_obs(subkey, self.U, self.Sigma, x0, ref, obs, cfg)
                else:
                    self.U, X, w = mppi_step_obs(subkey, self.U, x0, ref, obs, cfg)
            elif cfg.ADAPTIVE_COV:
                self.U, X, w, self.Sigma = mppi_step_cov(subkey, self.U, self.Sigma, x0, ref, cfg)
            else:
                self.U, X, w = mppi_step(subkey, self.U, x0, ref, cfg)
        # wait for the GPU here (float(self.U[0, 0]) below waits anyway), so the
        # wait is counted in the mppi segment instead of pub
        self.U.block_until_ready()
        t_d = time.perf_counter()

        # TODO: publish drive message: steering delta_0, speed v + a_0 * DT
        #       (float() turns a JAX array into a number)
        # copy U to the CPU once: every self.U[i, j] is its own GPU op (~5 ms each on the car)
        U_host = np.asarray(self.U)
        a0 = float(np.clip(U_host[0, 0], cfg.MIN_ACCEL, cfg.MAX_ACCEL))
        delta0 = float(np.clip(U_host[0, 1], cfg.MIN_STEER, cfg.MAX_STEER))
        drive = AckermannDriveStamped()
        drive.header.stamp = self.get_clock().now().to_msg()
        drive.header.frame_id = 'ego_racecar/base_link' if self.use_sim else 'base_link'
        drive.drive.steering_angle = delta0
        lead = float(self.get_parameter('accel_lead_time').value) if a0 > 0.0 else cfg.DT
        drive.drive.speed = float(np.clip(v + a0 * lead, cfg.MIN_SPEED, cfg.MAX_SPEED))
        # added: stuck recovery. MPPI only drives forward, so in a spot where every forward
        # trajectory collides (e.g. pointing at an obstacle in a tight turn) it just stops.
        # After stuck_time s standing still, back up for reverse_time s with the steering
        # opposite to what MPPI wants: reversing that way turns the nose toward the way out.
        if self.avoid and self.get_parameter('recovery').value:
            if self.recover_until is not None and t < self.recover_until:
                drive.drive.speed = float(self.get_parameter('reverse_speed').value)
                drive.drive.steering_angle = self.recover_steer
            else:
                if self.recover_until is not None:  # recovery just ended: plan from scratch
                    self.recover_until = None
                    self.U = jnp.zeros((cfg.H, cfg.NU))
                    self.cold = True
                if abs(v) < 0.05 and float(ref[0, 3]) > 0.1:
                    if self.stuck_since is None:
                        self.stuck_since = t
                    elif t - self.stuck_since > float(self.get_parameter('stuck_time').value):
                        self.recover_until = t + float(self.get_parameter('reverse_time').value)
                        self.recover_steer = cfg.MIN_STEER if delta0 >= 0.0 else cfg.MAX_STEER
                        self.stuck_since = None
                        self.get_logger().warn('stuck -> backing up')
                else:
                    self.stuck_since = None
        self.drive_pub.publish(drive)
        self.last_cmd = (a0, delta0)
        self.cmd_hist.append((time.monotonic(), a0, delta0))
        t_e = time.perf_counter()

        # odom stamp age at publish time (includes the particle filter's delay)
        age = self.get_clock().now().nanoseconds * 1e-9 - t
        if n_iter == 1:  # cold-start solves are logged on their own, not mixed into the stats
            for k, val in zip(self.seg, (t_a - t_start, t_b - t_a, t_c - t_b, t_o - t_c,
                                         t_d - t_o, t_e - t_d, age)):
                self.seg[k].append(val)
        else:
            self.get_logger().info(f'cold solve ({n_iter} iters): {1e3 * (t_d - t_o):.0f} ms')

        # timing (EMA) for latency compensation + a log line every 2 s
        dt_step = time.perf_counter() - t_start
        if n_iter == 1:  # cold starts run several iterations: not representative of the delay
            self.comp_time = dt_step if self.comp_time == 0.0 else 0.9 * self.comp_time + 0.1 * dt_step
        if t_start - self.last_log > 2.0:
            self.last_log = t_start
            ess = 1.0 / float(np.sum(np.asarray(w)[:, 0] ** 2))
            msg = (f'control step {1e3 * self.comp_time:.1f} ms, max gap {1e3 * self.max_gap:.0f} ms, '
                   f'v={v:.2f} m/s, ESS={ess:.0f}/{cfg.T}')
            if self.avoid:  # added: where the time goes
                msg += f', grid {1e3 * self.t_grid:.1f} ms, bend {1e3 * self.t_bend:.1f} ms'
            if self.max_gap > 0.1:  # the car drove blind for > 0.1 s: VM / visualization hiccup
                self.get_logger().warn(msg + '  <- hiccup')
            else:
                self.get_logger().info(msg)
            self.max_gap = 0.0
            stats = ' | '.join(f'{k} {1e3 * np.median(val):.1f}/{1e3 * np.max(val):.0f}'
                               for k, val in self.seg.items() if val)
            if stats:
                self.get_logger().info(f'p50/max ms: {stats}')
            for val in self.seg.values():
                val.clear()

        # TODO (optional, debugging only): visualize the reference, the trajectory
        #       of self.U and a few sampled trajectories, colored by their weight
        if self.get_parameter('visualize').value:
            self.publish_markers(odom_msg.header.frame_id or 'map', x0, ref, X, w)

    # ------------------------------------------------------------------ added
    def bend_ref(self, ref, px, py, yaw, obs, scan, cfg, v=0.0):
        """
        added: plan sideways offsets along the next bend_lookahead meters of the waypoints,
        starting from the car's own sideways position, and move each reference point by the
        offset at its distance along the path
        """
        gp = lambda k: self.get_parameter(k).value
        opts = dict(lookahead=float(gp('bend_lookahead')), extra=float(gp('bend_extra')),
                    lead_time=float(gp('bend_lead_time')), lead_dist=float(gp('bend_lead_dist')),
                    hold_blind=bool(gp('bend_hold_blind')), curvature=bool(gp('bend_curvature')),
                    curv_scale=float(gp('bend_curv_scale')), station=float(gp('bend_station')),
                    consistency=float(gp('bend_consistency')), map_prior=self.map_obs is not None)
        path, o_path, i0, self.bend_ok = detour_offsets(self.waypoints, px, py, yaw, v, obs,
                                                        (scan[0], scan[1]), self.lidar_offset, cfg, opts,
                                                        self.prev_bend)
        self.prev_bend = (i0, o_path)
        out = bend_reference(ref, path, o_path)
        self.last_bent_path = np.c_[path[:, 0] - o_path * np.sin(path[:, 2]),
                                    path[:, 1] + o_path * np.cos(path[:, 2])]
        return out

    def next_key(self):
        """added: the next of the pre-made random keys (numpy, no GPU op)"""
        key = self.keys[self.key_i % len(self.keys)]
        self.key_i += 1
        return key

    def scan_callback(self, msg):
        """added: keep the newest scan (ranges + beam angles) for the obstacle grid"""
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        angles = msg.angle_min + msg.angle_increment * np.arange(len(ranges))
        self.scan = (ranges, angles, float(msg.range_min))
        self.scan_seq += 1

    def map_callback(self, msg):
        """added (use_map_prior): the static map; from here on the obstacle grid uses MapObstacles"""
        gp = lambda k: float(self.get_parameter(k).value)
        self.map_obs = MapObstacles.from_occupancy_grid(msg, depth=gp('obstacle_depth'),
                                                        decay=gp('obstacle_decay'),
                                                        mem_range=gp('obstacle_range'))
        self.get_logger().info(f'map prior: {msg.info.width} x {msg.info.height} cells at '
                               f'{msg.info.resolution} m, {int(self.map_obs.wall.sum())} wall/unknown')

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
        if getattr(self, 'last_bent_path', None) is not None:  # added: planned detour (cyan)
            arr.markers.append(line(n_show + 2, self.last_bent_path,
                                    ColorRGBA(r=0.0, g=0.8, b=1.0, a=0.8), 0.05))
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
