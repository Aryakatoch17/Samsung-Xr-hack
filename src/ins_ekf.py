"""Fast strapdown INS + complementary attitude / bias observer."""

from __future__ import annotations

import numpy as np

from .math_utils import (
    G,
    gravity_aligned_quat,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rotmat_from_quat,
)


def severity_scale(t_frac: float) -> float:
    return 0.02 + 0.98 * (max(0.0, min(1.0, t_frac)) ** 1.7)


def estimate_biases_and_attitude(
    t_s: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    usable: np.ndarray,
    failed: np.ndarray,
    gap: np.ndarray,
    dup: np.ndarray,
    protect_s: float = 2.0,
) -> dict:
    """
    Pass-1 observer (no 15x15 matrices):
      - gyro integration + complementary gravity correction
      - slow accel/gyro bias tracking
    Returns histories needed for pass-2 reintegration.
    """
    n = len(t_s)
    ba = np.zeros((n, 3), dtype=np.float64)
    bg = np.zeros((n, 3), dtype=np.float64)
    quat = np.zeros((n, 4), dtype=np.float64)
    pos = np.zeros((n, 3), dtype=np.float64)
    vel = np.zeros((n, 3), dtype=np.float64)
    a_w_hist = np.zeros((n, 3), dtype=np.float64)
    w_hist = np.zeros((n, 3), dtype=np.float64)

    t0 = t_s[0]
    T = max(float(t_s[-1] - t0), 1e-6)
    prot = (t_s <= t0 + protect_s) & usable & ~failed
    init_idx = np.flatnonzero(prot)
    if init_idx.size < 5:
        init_idx = np.arange(min(50, n))

    bg0 = np.mean(gyro[init_idx], axis=0)
    ba0 = np.zeros(3)
    q = gravity_aligned_quat(np.mean(accel[init_idx], axis=0))
    p = np.zeros(3)
    v = np.zeros(3)
    g_w = np.array([0.0, 0.0, -G])

    last_a_w = np.zeros(3)
    last_w = np.zeros(3)
    grav_updates = 0

    # complementary gains (adapt with severity)
    kp_base = 0.4
    ba_alpha_base = 5e-4
    bg_alpha_base = 1e-4

    ba_i = ba0.copy()
    bg_i = bg0.copy()

    for i in range(n):
        if i == 0:
            ba[i], bg[i] = ba_i, bg_i
            quat[i] = q
            pos[i], vel[i] = p, v
            continue

        dt = t_s[i] - t_s[i - 1]
        if dup[i] or dt <= 0.0:
            ba[i], bg[i] = ba_i, bg_i
            quat[i] = q
            pos[i], vel[i] = p, v
            a_w_hist[i] = last_a_w
            w_hist[i] = last_w
            continue

        t_frac = (t_s[i] - t0) / T
        sev = severity_scale(t_frac)

        if failed[i]:
            q = quat_normalize(quat_mul(q, quat_from_rotvec(last_w * dt)))
            p = p + v * dt + 0.5 * last_a_w * dt * dt
            v = v + last_a_w * dt
            ba[i], bg[i] = ba_i, bg_i
            quat[i] = q
            pos[i], vel[i] = p, v
            a_w_hist[i] = last_a_w
            w_hist[i] = last_w
            continue

        if (gap[i] and dt > 0.05) or (not usable[i]):
            q = quat_normalize(quat_mul(q, quat_from_rotvec(last_w * dt)))
            p = p + v * dt + 0.5 * last_a_w * dt * dt
            v = v + last_a_w * dt
            ba[i], bg[i] = ba_i, bg_i
            quat[i] = q
            pos[i], vel[i] = p, v
            a_w_hist[i] = last_a_w
            w_hist[i] = last_w
            continue

        omega = gyro[i] - bg_i
        f = accel[i] - ba_i
        fn = np.linalg.norm(f)

        # Complementary gravity correction when specific force ~ g
        if abs(fn - G) < 2.0 and fn > 1.0:
            f_hat = f / fn
            # estimated gravity direction in body from attitude: R^T * [0,0,1]
            R = rotmat_from_quat(q)
            g_body = R.T @ np.array([0.0, 0.0, 1.0])
            err = np.cross(f_hat, g_body)  # rotate measured up toward estimated up
            kp = kp_base / (0.25 + sev)  # trust accel less as jamming grows
            omega = omega + kp * err
            # slow bias learning
            ba_i = ba_i + (ba_alpha_base / (0.3 + sev)) * (f - (fn * g_body))
            bg_i = bg_i - (bg_alpha_base / (0.3 + sev)) * err
            grav_updates += 1

        q = quat_normalize(quat_mul(q, quat_from_rotvec(omega * dt)))
        R = rotmat_from_quat(q)
        a_w = R @ f + g_w
        # mild clipping of absurd instantaneous world accel (spikes that leaked)
        an = np.linalg.norm(a_w)
        if an > 25.0:
            a_w = a_w * (25.0 / an)

        p = p + v * dt + 0.5 * a_w * dt * dt
        v = v + a_w * dt
        last_a_w = a_w
        last_w = omega

        ba[i], bg[i] = ba_i, bg_i
        quat[i] = q
        pos[i], vel[i] = p, v
        a_w_hist[i] = a_w
        w_hist[i] = omega

    return {
        "ba": ba,
        "bg": bg,
        "quat": quat,
        "pos": pos,
        "vel": vel,
        "a_w": a_w_hist,
        "omega": w_hist,
        "grav_updates": grav_updates,
        "q0": quat[0].copy(),
    }


def smooth_bias(bias: np.ndarray, usable: np.ndarray, win: int = 51) -> np.ndarray:
    n = len(bias)
    out = bias.copy()
    if n == 0:
        return out
    w = max(3, win if win % 2 == 1 else win + 1)
    half = w // 2
    filled = bias.copy()
    last = filled[0]
    for i in range(n):
        if usable[i]:
            last = filled[i]
        else:
            filled[i] = last
    last = filled[-1]
    for i in range(n - 1, -1, -1):
        if usable[i]:
            last = filled[i]
        else:
            filled[i] = last
    kernel = np.ones(w) / w
    for j in range(3):
        pad = np.pad(filled[:, j], (half, half), mode="edge")
        out[:, j] = np.convolve(pad, kernel, mode="valid")
    return out


def reintegrate(
    t_s: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    q0: np.ndarray,
    ba: np.ndarray,
    bg: np.ndarray,
    usable: np.ndarray,
    failed: np.ndarray,
    gap: np.ndarray,
    dup: np.ndarray,
    coast_a: np.ndarray,
    coast_w: np.ndarray,
    g_w: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if g_w is None:
        g_w = np.array([0.0, 0.0, -G])
    n = len(t_s)
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    quat = np.zeros((n, 4))
    q = quat_normalize(q0.copy())
    p = np.zeros(3)
    v = np.zeros(3)
    last_a_w = coast_a.copy()
    last_w = coast_w.copy()

    for i in range(n):
        if i == 0:
            quat[i] = q
            pos[i] = p
            vel[i] = v
            continue

        dt = t_s[i] - t_s[i - 1]
        if dup[i] or dt <= 0.0:
            quat[i] = q
            pos[i] = p
            vel[i] = v
            continue

        if failed[i]:
            q = quat_normalize(quat_mul(q, quat_from_rotvec(coast_w * dt)))
            p = p + v * dt + 0.5 * coast_a * dt * dt
            v = v + coast_a * dt
            quat[i] = q
            pos[i] = p
            vel[i] = v
            continue

        if (gap[i] and dt > 0.05) or (not usable[i]):
            q = quat_normalize(quat_mul(q, quat_from_rotvec(last_w * dt)))
            p = p + v * dt + 0.5 * last_a_w * dt * dt
            v = v + last_a_w * dt
            quat[i] = q
            pos[i] = p
            vel[i] = v
            continue

        omega = gyro[i] - bg[i]
        acc_b = accel[i] - ba[i]
        q = quat_normalize(quat_mul(q, quat_from_rotvec(omega * dt)))
        R = rotmat_from_quat(q)
        a_w = R @ acc_b + g_w
        an = np.linalg.norm(a_w)
        if an > 25.0:
            a_w = a_w * (25.0 / an)
        p = p + v * dt + 0.5 * a_w * dt * dt
        v = v + a_w * dt
        last_a_w = a_w
        last_w = omega
        quat[i] = q
        pos[i] = p
        vel[i] = v

    return pos, vel, quat
