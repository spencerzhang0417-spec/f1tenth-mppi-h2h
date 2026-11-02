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
import os
import time
import re
from rclpy.qos import QoSProfile
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from scipy.ndimage import distance_transform_edt


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
    # obstacle avoidance from /scan (only used when avoid_obstacles is True)
    OBS_RES: float = 0.05      # [m] cell size of the local grid built from the scan
    OBS_CELLS: int = 280       # grid is OBS_CELLS x OBS_CELLS around the car (280 * 0.05 = 14 m)
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


def scan_to_grid(ranges, angles, range_min, px, py, yaw, lidar_offset, cfg, blind_radius=0.8):
    """
    Added: LaserScan -> local clearance grid around the car, axis-aligned in the map frame.
    Only cells the scan has actually seen as empty (between the laser and each return) count
    as free. Everything else -- the returns, the inside and far side of obstacles, the space
    behind walls -- counts as blocked, so a gap that only looks open because its far side was
    never seen is not used. The laser's rear blind spot within blind_radius of the car is free.
    :return: (dist (OBS_CELLS, OBS_CELLS) float32, ox, oy), same layout as obstacle_cost expects
    """
    n, res = cfg.OBS_CELLS, cfg.OBS_RES
    half = 0.5 * n * res
    ox, oy = px - half, py - half
    if len(ranges) == 0:  # no scan (e.g. the warm-up compile): everything free
        return (jnp.full((n, n), 10.0, dtype=jnp.float32), jnp.float32(ox), jnp.float32(oy))
    lx0 = px + lidar_offset * np.cos(yaw)
    ly0 = py + lidar_offset * np.sin(yaw)
    r_max = 1.5 * half
    hit = np.isfinite(ranges) & (ranges > range_min) & (ranges < r_max)
    r_free = np.where(hit, ranges, r_max)
    r_free = np.where(np.isfinite(r_free), r_free, r_max)

    def to_cells(r, a):
        wx = lx0 + r * np.cos(yaw + a)
        wy = ly0 + r * np.sin(yaw + a)
        col = np.floor((wx - ox) / res).astype(int)
        row = np.floor((wy - oy) / res).astype(int)
        keep = (row >= 0) & (row < n) & (col >= 0) & (col < n)
        return row[keep], col[keep]

    free = np.zeros((n, n), dtype=bool)
    # free space: points every res/2 along each beam up to the return (the return's own cell
    # is set back to blocked below; stopping a whole cell short would shrink every gap by 0.1 m)
    t = np.arange(0.0, r_max, 0.5 * res)                                    # (S,)
    along = t[None, :] < (r_free[:, None] - 0.25 * res)                     # (B, S)
    rr, cc = to_cells(np.broadcast_to(t, along.shape)[along],
                      np.broadcast_to(angles[:, None], along.shape)[along])
    free[rr, cc] = True
    # rear blind spot of the laser (outside its field of view) close to the car
    yy, xx = np.mgrid[0:n, 0:n]
    cx, cy = (xx + 0.5) * res + ox - px, (yy + 0.5) * res + oy - py
    b = np.arctan2(cy, cx) - yaw
    b = np.arctan2(np.sin(b), np.cos(b))
    free |= (np.hypot(cx, cy) < blind_radius) & ((b < angles[0]) | (b > angles[-1]))
    # the returns themselves are blocked
    rr, cc = to_cells(ranges[hit], angles[hit])
    free[rr, cc] = False
    dist = distance_transform_edt(free) * res  # distance of every cell to the nearest blocked one
    return (jnp.asarray(dist, dtype=jnp.float32), jnp.float32(ox), jnp.float32(oy))


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
    ok = np.isfinite(ranges) & (ranges > range_min)

    def lin(sel, d0, d1):
        if not sel.any():
            return 1.0
        d = float(np.min(ranges[sel]))
        return float(np.clip(f_min + (1.0 - f_min) * (d - d0) / (d1 - d0), f_min, 1.0))

    f_front = lin(ok & (np.abs(angles) < front_half_angle), front_stop, front_free)
    f_side = lin(ok & (np.abs(angles) < 0.5 * np.pi), side_stop, side_free)
    return min(f_front, f_side)


def plan_lateral_offsets(path, obs, cfg, need, scan_view=None, start_offset=0.0,
                         max_offset=1.0, step=0.05, w_smooth=2.0):
    """
    Added: bend the reference around obstacles. For the next few meters of the waypoint path
    (path: (M, 4) [x, y, yaw, v], ~0.1 m apart) pick a sideways offset at every point by dynamic
    programming over a grid of offsets, so that every point keeps >= need clearance in the scan
    grid and the offset changes by at most one grid step (step) per point (a smooth curve the car
    can follow). One offset for the whole horizon cannot pass a slalom; a per-point one can.
      cost = sum |offset| + w_smooth * |change| ; blocked cells cost 1e3 (soft, so DP always ends)
    A shifted point must also be in front of the scan return in its direction (not behind a wall),
    unless no offset at that point is visible at all (e.g. around a corner): then the grid decides.
    :return: offsets (M,)
    """
    dist = np.asarray(obs[0]); ox, oy = float(obs[1]), float(obs[2])
    n = dist.shape[0]
    offs = np.arange(-max_offset, max_offset + 1e-9, step)                   # (K,)
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
    node = np.abs(offs)[None, :] + np.where(free, 0.0, 1e3)                  # (M, K)
    k0 = int(np.argmin(np.abs(offs - start_offset)))
    INF = 1e12
    J = np.full(K, INF); J[k0] = node[0, k0]
    back = np.zeros((M, K), dtype=int)
    for m in range(1, M):
        best = np.full(K, INF); arg = np.zeros(K, dtype=int)
        for dk in (-1, 0, 1):                                                # max one step per point
            prev = np.roll(J, dk)
            if dk == 1:
                prev[0] = INF
            elif dk == -1:
                prev[-1] = INF
            cand = prev + w_smooth * step * abs(dk)
            better = cand < best
            best[better] = cand[better]
            arg[better] = (np.arange(K) - dk)[better]
        J = best + node[m]
        back[m] = arg
    k = int(np.argmin(J))
    out = np.empty(M)
    for m in range(M - 1, -1, -1):
        out[m] = offs[k]
        k = back[m, k]
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
        self.declare_parameter('num_samples', int(MPPIConfig.T))  # T: lower it if the car's CPU is slow
        self.declare_parameter('visualize', False)  # rollouts in RViz: debugging / sim video only
        # latency compensation: predict the state forward by the control delay before planning.
        # -1.0 = automatic (measured time of one control step), 0.0 = off, > 0 = fixed delay [s]
        self.declare_parameter('latency_comp', -1.0)
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
        self.key = jax.random.PRNGKey(0)

        # jax.jit compiles mppi_step the first time it is called, which takes a
        # few seconds: call it once here so the car does not wait for it
        mppi_step(self.key, self.U, np.zeros(self.config.NX),
                  np.zeros((self.config.H + 1, self.config.NX)), self.config)

        # added: ROS-parameter overrides of the config (new config -> compile once more)
        overrides = dict(MAX_SPEED=float(self.get_parameter('max_speed').value),
                         T=int(self.get_parameter('num_samples').value),
                         Q=tuple(float(q) for q in self.get_parameter('Q').value),
                         Qf=tuple(float(q) for q in self.get_parameter('Qf').value),
                         R=tuple(float(q) for q in self.get_parameter('R').value),
                         Rd=tuple(float(q) for q in self.get_parameter('Rd').value),
                         LAMBDA=float(self.get_parameter('lambda').value),
                         U_STD=tuple(float(q) for q in self.get_parameter('u_std').value))
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
        # shift the reference sideways when it runs through an obstacle (avoids getting stuck)
        self.declare_parameter('shift_ref', True)
        self.declare_parameter('bend_lookahead', 4.0)  # [m] of path the offsets are planned over
        self.declare_parameter('bend_extra', 0.03)     # [m] bent path keeps obs_radius + this clearance
        # stuck recovery: back up when standing still for stuck_time s
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
            empty = scan_to_grid(np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0, cfg)
            z_x, z_ref = np.zeros(cfg.NX), np.zeros((cfg.H + 1, cfg.NX))
            mppi_step(self.key, self.U, z_x, z_ref, cfg)  # compile now, not while driving
            if cfg.ADAPTIVE_COV:
                mppi_step_cov_obs(self.key, self.U, self.Sigma, z_x, z_ref, empty, cfg)
            else:
                mppi_step_obs(self.key, self.U, z_x, z_ref, empty, cfg)
            self.get_logger().info(f'obstacle avoidance ON: {self.get_parameter("scan_topic").value}, '
                                   f'lidar_offset {self.lidar_offset} m, steer_std {cfg.U_STD[1]}, '
                                   f'obs_radius {cfg.OBS_RADIUS}, obs_margin {cfg.OBS_MARGIN}')

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

        # added: local clearance grid from the newest scan, placed with the measured pose
        # (not the latency-compensated x0: the scan was taken where the car is now)
        obs = None
        if self.avoid and self.scan is not None:
            ranges, angles, range_min = self.scan
            obs = scan_to_grid(ranges, angles, range_min, p.x, p.y, yaw, self.lidar_offset, cfg)
            if self.get_parameter('shift_ref').value:
                ref = self.bend_ref(np.asarray(ref, dtype=float), p.x, p.y, yaw, obs,
                                    (ranges, angles), cfg)
            if self.get_parameter('slow_near_obstacles').value:
                f = obstacle_speed_factor(ranges, angles, range_min,
                                          f_min=float(self.get_parameter('slow_min_factor').value))
                ref = np.array(ref, dtype=float)
                ref[:, 3] *= f  # ref columns: x, y, yaw, v

        # a new random key at every step, so MPPI draws new samples
        self.key, subkey = jax.random.split(self.key)
        # TODO: run MPPI, keep the new mean control sequence in self.U
        # From a cold start (U = 0) a single update is noisy, so iterate a few times
        n_iter = cfg.N_ITER_COLD if self.cold else cfg.N_ITER
        self.cold = False
        for it in range(n_iter):
            if it > 0:
                self.key, subkey = jax.random.split(self.key)
            if obs is not None:  # added: same update plus the obstacle cost
                if cfg.ADAPTIVE_COV:
                    self.U, X, w, self.Sigma = mppi_step_cov_obs(subkey, self.U, self.Sigma, x0, ref, obs, cfg)
                else:
                    self.U, X, w = mppi_step_obs(subkey, self.U, x0, ref, obs, cfg)
            elif cfg.ADAPTIVE_COV:
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
                        self.recover_steer = -cfg.MAX_STEER if delta0 >= 0.0 else cfg.MAX_STEER
                        self.stuck_since = None
                        self.get_logger().warn('stuck -> backing up')
                else:
                    self.stuck_since = None
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
    def bend_ref(self, ref, px, py, yaw, obs, scan, cfg):
        """
        added: plan sideways offsets along the next bend_lookahead meters of the waypoints,
        starting from the car's own sideways position, and move each reference point by the
        offset at its distance along the path
        """
        wps = self.waypoints
        N = len(wps)
        i0 = int(np.argmin(np.hypot(wps[:, 0] - px, wps[:, 1] - py)))
        ds = float(np.mean(np.hypot(np.diff(wps[:50, 0]), np.diff(wps[:50, 1]))))
        M = max(int(float(self.get_parameter('bend_lookahead').value) / ds), 5)
        path = wps[(i0 + np.arange(M)) % N]
        # the car's own sideways offset from the path, so the bent path starts at the car
        nx, ny = -np.sin(path[0, 2]), np.cos(path[0, 2])
        start = (px - path[0, 0]) * nx + (py - path[0, 1]) * ny
        lx = px + self.lidar_offset * np.cos(yaw)
        ly = py + self.lidar_offset * np.sin(yaw)
        need = cfg.OBS_RADIUS + float(self.get_parameter('bend_extra').value)
        o_path = plan_lateral_offsets(path, obs, cfg, need=need,
                                      scan_view=(scan[0], scan[1], lx, ly, yaw), start_offset=start)
        # distance of every reference point along the path, measured from path[0]
        tx, ty = np.cos(path[0, 2]), np.sin(path[0, 2])
        s0 = (ref[0, 0] - path[0, 0]) * tx + (ref[0, 1] - path[0, 1]) * ty
        s_ref = s0 + np.r_[0.0, np.cumsum(np.hypot(np.diff(ref[:, 0]), np.diff(ref[:, 1])))]
        o_ref = np.interp(s_ref, np.arange(M) * ds, o_path)
        out = np.array(ref, dtype=float)
        out[:, 0] += o_ref * -np.sin(ref[:, 2])
        out[:, 1] += o_ref * np.cos(ref[:, 2])
        self.last_bent_path = np.c_[path[:, 0] - o_path * np.sin(path[:, 2]),
                                    path[:, 1] + o_path * np.cos(path[:, 2])]
        return out

    def scan_callback(self, msg):
        """added: keep the newest scan (ranges + beam angles) for the obstacle grid"""
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        angles = msg.angle_min + msg.angle_increment * np.arange(len(ranges))
        self.scan = (ranges, angles, float(msg.range_min))

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