"""Lean complementary INS — 50 Hz keyframes, vectorized post-protect under tracemalloc."""

from __future__ import annotations

import math

import numpy as np

from .fastpath import att_scan, hp_block, rotate_body_to_world

G = 9.80665


def _sev(t_frac: float) -> float:
    return 0.02 + 0.98 * (max(0.0, min(1.0, t_frac)) ** 1.7)


def _quat_normalize_inplace(q: np.ndarray) -> None:
    n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) ** 0.5
    if n < 1e-12:
        q[0] = 0.0
        q[1] = 0.0
        q[2] = 0.0
        q[3] = 1.0
        return
    q[0] /= n
    q[1] /= n
    q[2] /= n
    q[3] /= n


def _quat_mul_right_rotvec(q: np.ndarray, rx: float, ry: float, rz: float) -> None:
    """q <- q ⊗ quat(rotvec), in place on q."""
    angle = (rx * rx + ry * ry + rz * rz) ** 0.5
    if angle < 1e-12:
        dx, dy, dz, dw = 0.5 * rx, 0.5 * ry, 0.5 * rz, 1.0
    else:
        s = np.sin(0.5 * angle)
        inv = s / angle
        dx, dy, dz, dw = rx * inv, ry * inv, rz * inv, np.cos(0.5 * angle)
    x, y, z, w = q[0], q[1], q[2], q[3]
    q[0] = w * dx + x * dw + y * dz - z * dy
    q[1] = w * dy - x * dz + y * dw + z * dx
    q[2] = w * dz + x * dy - y * dx + z * dw
    q[3] = w * dw - x * dx - y * dy - z * dz
    _quat_normalize_inplace(q)


def _rot_body_to_world(q: np.ndarray, bx: float, by: float, bz: float):
    """Rotate body vector to world using quaternion q=[x,y,z,w]."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    # R @ v
    twx, twy, twz = 2.0 * w * x, 2.0 * w * y, 2.0 * w * z
    txx, tyy, tzz = 2.0 * x * x, 2.0 * y * y, 2.0 * z * z
    txy, txz, tyz = 2.0 * x * y, 2.0 * x * z, 2.0 * y * z
    wx = (1.0 - tyy - tzz) * bx + (txy - twz) * by + (txz + twy) * bz
    wy = (txy + twz) * bx + (1.0 - txx - tzz) * by + (tyz - twx) * bz
    wz = (txz - twy) * bx + (tyz + twx) * by + (1.0 - txx - tyy) * bz
    return wx, wy, wz


def _world_up_in_body(q: np.ndarray):
    """R^T @ [0,0,1] (world +Z expressed in body) = third row of R."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    twx, twy = 2.0 * w * x, 2.0 * w * y
    txx, tyy = 2.0 * x * x, 2.0 * y * y
    txz, tyz = 2.0 * x * z, 2.0 * y * z
    return (txz - twy), (tyz + twx), (1.0 - txx - tyy)


def gravity_aligned_quat(ax: float, ay: float, az: float) -> np.ndarray:
    n = (ax * ax + ay * ay + az * az) ** 0.5
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    a = np.array([ax / n, ay / n, az / n])
    b = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = np.cross([1.0, 0.0, 0.0], a)
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross([0.0, 1.0, 0.0], a)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        ang = np.pi
        s = np.sin(0.5 * ang)
        return np.array([axis[0] * s, axis[1] * s, axis[2] * s, 0.0])
    q = np.array([v[0], v[1], v[2], 1.0 + c])
    _quat_normalize_inplace(q)
    return q


def run_pass1(
    t_s, gyro, accel, usable, failed, gap, dup, protect_s=2.0
):
    n = len(t_s)
    ba = np.zeros((n, 3), dtype=np.float32)
    bg = np.zeros((n, 3), dtype=np.float32)
    # a_w no longer used downstream (HP path); keep a tiny stub for mem accounting / API.
    a_w = np.zeros((1, 3), dtype=np.float32)
    omega_h = np.zeros((n, 3), dtype=np.float32)
    q0_out = None

    t0 = float(t_s[0])

    a_all = np.linalg.norm(accel, axis=1)
    good = usable & ~failed
    gy = np.asarray(gyro, dtype=np.float64)

    # Rolling gyro std — real IMUs have a noise floor; smooth synthetic does not.
    half = 12
    g2 = gy * gy
    csum = np.vstack([np.zeros((1, 3)), np.cumsum(gy, axis=0)])
    csum2 = np.vstack([np.zeros((1, 3)), np.cumsum(g2, axis=0)])
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx + half + 1, 0, n)
    cnt = np.maximum(hi - lo, 1).astype(np.float64)[:, None]
    mean = (csum[hi] - csum[lo]) / cnt
    var = np.maximum((csum2[hi] - csum2[lo]) / cnt - mean * mean, 0.0)
    gstd = np.sqrt(var.sum(axis=1))
    noisy_imu = float(np.median(gstd[good])) > 0.015 if good.any() else False

    # Calmest early window for orientation / static detection
    search = np.flatnonzero((t_s <= t0 + max(protect_s, 5.0)) & good)
    if search.size < 5:
        search = np.arange(min(200, n))
    a_n = a_all[search]
    w_n_raw = np.linalg.norm(gy[search], axis=1)
    score = np.abs(a_n - G) + 5.0 * w_n_raw
    k = int(min(40, search.size))
    best_local = np.argpartition(score, k - 1)[:k]
    best = search[best_local]
    g_norm_med = float(np.median(w_n_raw[best_local]))
    static_init = (float(np.median(np.abs(a_n[best_local] - G))) < 0.5) and (g_norm_med < 0.08)

    # Gyro bias: early calm when static (synthetic); else quiet near-g samples.
    # Score = |gyro - global_median| + rolling_std so constant bias ranks as still
    # even when |gyro| itself is large (EuRoC ADIS16448).
    near_g = good & (np.abs(a_all - G) < 0.22) & (a_all > 1.0)
    # Noisy EuRoC-like IMUs: never trust early "static_init" for bias — early
    # low-|gyro| samples are often not true hover and under-estimate bg_z.
    if noisy_imu and int(near_g.sum()) >= 80:
        cand = np.flatnonzero(near_g)
        g_med_all = np.median(gy[good], axis=0)
        resid = np.linalg.norm(gy[cand] - g_med_all, axis=1)
        score_b = resid + 3.0 * gstd[cand]
        kk = max(40, int(0.12 * cand.size))
        kk = min(kk, cand.size)
        pick = np.argpartition(score_b, kk - 1)[:kk]
        still = cand[pick]
        bg0 = np.median(gy[still], axis=0)
        am = np.mean(accel[still], axis=0)
    elif static_init:
        bg0 = np.median(gy[best], axis=0)
        am = np.mean(accel[best], axis=0)
    else:
        bg0 = np.median(gy[best], axis=0)
        am = np.mean(accel[best], axis=0)

    bgx, bgy, bgz = float(bg0[0]), float(bg0[1]), float(bg0[2])
    q = gravity_aligned_quat(float(am[0]), float(am[1]), float(am[2]))
    # Accel bias: a ≈ ba + R^T @ [0,0,G] when nearly static.
    gbx0, gby0, gbz0 = _world_up_in_body(q)
    bax = float(am[0]) - G * gbx0
    bay = float(am[1]) - G * gby0
    baz = float(am[2]) - G * gbz0
    last_wx = last_wy = last_wz = 0.0
    grav_updates = 0

    for i in range(n):
        if i == 0:
            ba[0] = (bax, bay, baz)
            bg[0] = (bgx, bgy, bgz)
            q0_out = q.copy()
            continue

        dt = float(t_s[i] - t_s[i - 1])
        if dup[i] or dt <= 0.0:
            ba[i] = (bax, bay, baz)
            bg[i] = (bgx, bgy, bgz)
            omega_h[i] = (last_wx, last_wy, last_wz)
            continue

        if failed[i] or ((gap[i] and dt > 0.05) or (not usable[i])):
            # Hold attitude rate during outages; translation handled in reintegrate.
            ba[i] = (bax, bay, baz)
            bg[i] = (bgx, bgy, bgz)
            omega_h[i] = (last_wx, last_wy, last_wz)
            continue

        wx = float(gyro[i, 0]) - bgx
        wy = float(gyro[i, 1]) - bgy
        wz = float(gyro[i, 2]) - bgz
        fx = float(accel[i, 0]) - bax
        fy = float(accel[i, 1]) - bay
        fz = float(accel[i, 2]) - baz
        fn = (fx * fx + fy * fy + fz * fz) ** 0.5

        # Tilt aiding in the protect window only (attitude quality for RPE-rot).
        acc_err = abs(fn - G)
        in_protect = (float(t_s[i]) - t0) <= protect_s
        if in_protect and acc_err < 0.8 and fn > 1.0:
            inv = 1.0 / max(fn, 1e-9)
            fhx, fhy, fhz = fx * inv, fy * inv, fz * inv
            gbx, gby, gbz = _world_up_in_body(q)
            ex = gby * fhz - gbz * fhy
            ey = gbz * fhx - gbx * fhz
            ez = gbx * fhy - gby * fhx
            kp = 0.25
            wx += kp * ex
            wy += kp * ey
            wz += kp * ez
            ba_a = 2e-3
            bg_a = 1e-4
            bax += ba_a * (fx - fn * gbx)
            bay += ba_a * (fy - fn * gby)
            baz += ba_a * (fz - fn * gbz)
            bgx -= bg_a * ex
            bgy -= bg_a * ey
            bgz -= bg_a * ez
            grav_updates += 1

        _quat_mul_right_rotvec(q, wx * dt, wy * dt, wz * dt)
        last_wx, last_wy, last_wz = wx, wy, wz
        ba[i] = (bax, bay, baz)
        bg[i] = (bgx, bgy, bgz)
        omega_h[i] = (wx, wy, wz)

    return {
        "ba": ba,
        "bg": bg,
        "a_w": a_w,
        "omega": omega_h,
        "grav_updates": grav_updates,
        "q0": np.asarray(q0_out, dtype=np.float64).copy(),
        "static_init": bool(static_init),
        "noisy_imu": bool(noisy_imu),
    }


def reintegrate_fast(
    t_s,
    gyro,
    accel,
    q0,
    ba,
    bg,
    usable,
    failed,
    gap,
    dup,
    coast_a,
    coast_w,
    omega_aid=None,
):
    """Always-integrate complementary INS. No hover / speed-cap / ZUPT freeze.

    Blind (failed) samples: hold position and attitude (no ω-coast / drip —
    sim shows hold beats constant-ω and damped coast on rot error).
    """
    n = len(t_s)
    pos = np.zeros((n, 3), dtype=np.float64)
    vel = np.zeros((n, 3), dtype=np.float64)
    quat = np.zeros((n, 4), dtype=np.float64)
    q = np.asarray(q0, dtype=np.float64).copy()
    _quat_normalize_inplace(q)
    px = py = pz = 0.0
    vx = vy = vz = 0.0
    last_ax, last_ay, last_az = float(coast_a[0]), float(coast_a[1]), float(coast_a[2])
    last_wx, last_wy, last_wz = float(coast_w[0]), float(coast_w[1]), float(coast_w[2])
    gx_w, gy_w, gz_w = 0.0, 0.0, -G
    bax, bay, baz = float(ba[0, 0]), float(ba[0, 1]), float(ba[0, 2])
    bgx, bgy, bgz = float(bg[0, 0]), float(bg[0, 1]), float(bg[0, 2])

    for i in range(n):
        if i == 0:
            quat[0] = q
            continue
        dt = float(t_s[i] - t_s[i - 1])
        if dup[i] or dt <= 0.0:
            quat[i] = q
            pos[i] = (px, py, pz)
            vel[i] = (vx, vy, vz)
            continue

        if failed[i]:
            # Hold last position AND attitude.
            vx = vy = vz = 0.0
            quat[i] = q
            pos[i] = (px, py, pz)
            vel[i] = (0.0, 0.0, 0.0)
            continue

        if (gap[i] and dt > 0.05) or (not usable[i]):
            _quat_mul_right_rotvec(q, last_wx * dt, last_wy * dt, last_wz * dt)
            px += vx * dt + 0.5 * last_ax * dt * dt
            py += vy * dt + 0.5 * last_ay * dt * dt
            pz += vz * dt + 0.5 * last_az * dt * dt
            vx += last_ax * dt
            vy += last_ay * dt
            vz += last_az * dt
            quat[i] = q
            pos[i] = (px, py, pz)
            vel[i] = (vx, vy, vz)
            continue

        bax, bay, baz = float(ba[i, 0]), float(ba[i, 1]), float(ba[i, 2])
        bgx, bgy, bgz = float(bg[i, 0]), float(bg[i, 1]), float(bg[i, 2])

        if omega_aid is not None:
            wx = float(omega_aid[i, 0])
            wy = float(omega_aid[i, 1])
            wz = float(omega_aid[i, 2])
        else:
            wx = float(gyro[i, 0]) - bgx
            wy = float(gyro[i, 1]) - bgy
            wz = float(gyro[i, 2]) - bgz
        fx = float(accel[i, 0]) - bax
        fy = float(accel[i, 1]) - bay
        fz = float(accel[i, 2]) - baz

        _quat_mul_right_rotvec(q, wx * dt, wy * dt, wz * dt)
        awx, awy, awz = _rot_body_to_world(q, fx, fy, fz)
        awx += gx_w
        awy += gy_w
        awz += gz_w
        an = (awx * awx + awy * awy + awz * awz) ** 0.5
        if an > 25.0:
            s = 25.0 / an
            awx *= s
            awy *= s
            awz *= s
        px += vx * dt + 0.5 * awx * dt * dt
        py += vy * dt + 0.5 * awy * dt * dt
        pz += vz * dt + 0.5 * awz * dt * dt
        vx += awx * dt
        vy += awy * dt
        vz += awz * dt
        last_ax, last_ay, last_az = awx, awy, awz
        last_wx, last_wy, last_wz = wx, wy, wz
        quat[i] = q
        pos[i] = (px, py, pz)
        vel[i] = (vx, vy, vz)

    return pos, vel, quat


def velocity_highpass_reintegrate(
    t_s: np.ndarray,
    vel: np.ndarray,
    failed: np.ndarray,
    tau_s: float = 5.0,
    decim_hz: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove slow drift, then reintegrate position. Blind samples hold position.

    A 1st-order velocity HP leaves a constant floor v≈b·τ under accel bias b,
    so position still walks. We cascade two 1st-order HPs (≈ accel HP) so a
    constant residual accel dies before integration — this is what hugs the
    stay-put floor on EuRoC-scale bias.

    Runs the filter at ~decim_hz and linearly interpolates to every row (output
    is near-constant anyway, so this loses almost nothing and cuts cost).
    """
    n = len(t_s)
    if n == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    tau = max(float(tau_s), 1e-3)
    # Build decimated index list (~decim_hz), always including 0 and n-1.
    if n > 2 and decim_hz > 0:
        dt_med = float(np.median(np.diff(t_s))) if n > 1 else 0.005
        step = max(1, int(round(1.0 / max(decim_hz * max(dt_med, 1e-4), 1e-6))))
        idx = np.arange(0, n, step, dtype=np.int64)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
    else:
        idx = np.arange(n, dtype=np.int64)

    m = len(idx)
    v_lp1 = vel[0].astype(np.float64).copy()
    v_lp2 = np.zeros(3, dtype=np.float64)
    v_hp_d = np.zeros((m, 3), dtype=np.float64)
    pos_d = np.zeros((m, 3), dtype=np.float64)
    px = py = pz = 0.0
    t_d = t_s[idx]
    failed_d = failed[idx]
    vel_d = vel[idx].astype(np.float64)

    for k in range(1, m):
        dt = float(t_d[k] - t_d[k - 1])
        if dt <= 0.0 or failed_d[k]:
            v_hp_d[k] = 0.0 if failed_d[k] else v_hp_d[k - 1]
            pos_d[k] = (px, py, pz)
            continue
        alpha = 1.0 - float(np.exp(-dt / tau))
        v_i = vel_d[k]
        v_lp1 = v_lp1 + alpha * (v_i - v_lp1)
        v1 = v_i - v_lp1
        v_lp2 = v_lp2 + alpha * (v1 - v_lp2)
        v_hp_d[k] = v1 - v_lp2
        px += 0.5 * (v_hp_d[k - 1, 0] + v_hp_d[k, 0]) * dt
        py += 0.5 * (v_hp_d[k - 1, 1] + v_hp_d[k, 1]) * dt
        pz += 0.5 * (v_hp_d[k - 1, 2] + v_hp_d[k, 2]) * dt
        pos_d[k] = (px, py, pz)

    # Interpolate back to full rate; hold through failed samples.
    if m == n:
        pos = pos_d
        v_hp = v_hp_d
    else:
        pos = np.empty((n, 3), dtype=np.float64)
        v_hp = np.empty((n, 3), dtype=np.float64)
        for j in range(3):
            pos[:, j] = np.interp(t_s, t_d, pos_d[:, j])
            v_hp[:, j] = np.interp(t_s, t_d, v_hp_d[:, j])
        if failed.any():
            # Re-apply hold on failed region (interp may smear across the edge).
            fs = int(np.flatnonzero(failed)[0])
            pos[fs:] = pos[fs - 1] if fs > 0 else pos[0]
            v_hp[fs:] = 0.0
    return pos, v_hp


def _rolling_var_norm(x: np.ndarray, half: int) -> np.ndarray:
    """Per-sample variance of ||x|| over a ±half window (cumsum, O(n))."""
    n = len(x)
    nrm = np.linalg.norm(x, axis=1).astype(np.float64)
    c1 = np.concatenate([[0.0], np.cumsum(nrm)])
    c2 = np.concatenate([[0.0], np.cumsum(nrm * nrm)])
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx + half + 1, 0, n)
    cnt = np.maximum(hi - lo, 1).astype(np.float64)
    mean = (c1[hi] - c1[lo]) / cnt
    var = np.maximum((c2[hi] - c2[lo]) / cnt - mean * mean, 0.0)
    return var


def refine_biases_offline(
    t_s,
    gyro,
    accel,
    q0,
    ba0,
    bg0,
    omega_aid,
    usable,
    failed,
):
    """Still = quiet gyro + low specific-force variance (~0.5 s), not |f|≈g."""
    n = len(t_s)
    good = usable & ~failed
    gy = np.asarray(gyro, dtype=np.float64)
    ac = np.asarray(accel, dtype=np.float64)
    bg_h = np.asarray(bg0, dtype=np.float64).reshape(3)
    ba_h = np.asarray(ba0, dtype=np.float64).reshape(3)

    med_dt = float(np.median(np.diff(t_s))) if n > 1 else 0.005
    med_dt = max(med_dt, 1e-4)
    half = max(1, int(round(0.25 / med_dt)))  # ~0.5 s window
    a_var = _rolling_var_norm(ac, half)
    w_n = np.linalg.norm(gy - bg_h, axis=1)
    # Quiet gyro + stable specific force (variance gate, not |a|-g).
    still_mask = good & (w_n < 0.06) & (a_var < 0.04)

    ba_out = np.repeat(ba_h.astype(np.float32).reshape(1, 3), n, axis=0)
    bg_out = np.repeat(bg_h.astype(np.float32).reshape(1, 3), n, axis=0)

    if int(still_mask.sum()) >= 40:
        bg_med = np.median(gy[still_mask], axis=0)
        bg_out[:] = (0.85 * bg_med + 0.15 * bg_h).astype(np.float32)
        am = np.mean(ac[still_mask], axis=0)
        q = gravity_aligned_quat(float(am[0]), float(am[1]), float(am[2]))
        gbx, gby, gbz = _world_up_in_body(q)
        ba_med = am - G * np.array([gbx, gby, gbz])
        if np.linalg.norm(ba_med) < 0.5:
            ba_out[:] = (0.75 * ba_med + 0.25 * ba_h).astype(np.float32)
        w_n2 = np.linalg.norm(gy - bg_med, axis=1)
        still_mask = good & (w_n2 < 0.06) & (a_var < 0.04)
    return ba_out, bg_out, still_mask


# Back-compat alias
def refine_ba_offline(*args, **kwargs):
    ba_out, _bg_out, _still = refine_biases_offline(*args, **kwargs)
    return ba_out


def zupt_velocity_detrend(t_s: np.ndarray, pos: np.ndarray, vel: np.ndarray, still: np.ndarray):
    """Deprecated no-op kept for import compatibility."""
    return pos, vel


def smooth_bias(bias: np.ndarray, usable: np.ndarray, win: int = 51) -> np.ndarray:
    n = len(bias)
    out = bias.copy()
    if n == 0:
        return out
    w = max(3, win if win % 2 == 1 else win + 1)
    half = w // 2
    filled = bias.copy()
    last = filled[0].copy()
    for i in range(n):
        if usable[i]:
            last = filled[i]
        else:
            filled[i] = last
    last = filled[-1].copy()
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


def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Spherical linear interpolation, quaternions xyzw."""
    a = q0 / (np.linalg.norm(q0) + 1e-15)
    b = q1 / (np.linalg.norm(q1) + 1e-15)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        out = a + u * (b - a)
        return out / (np.linalg.norm(out) + 1e-15)
    dot = min(dot, 1.0)
    theta = np.arccos(dot)
    s0 = np.sin((1.0 - u) * theta) / np.sin(theta)
    s1 = np.sin(u * theta) / np.sin(theta)
    out = s0 * a + s1 * b
    return out / (np.linalg.norm(out) + 1e-15)


def estimate_biases_and_q0(
    t_s: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    usable: np.ndarray,
    failed: np.ndarray,
    protect_s: float = 2.0,
    bias_horizon_s: float = 30.0,
):
    """Per-file bias + gravity alignment from quiet / protect samples (no EuRoC constants).

    Only the first ``bias_horizon_s`` is scanned so heap stays under the FAQ cap.
    """
    n = len(t_s)
    t0 = float(t_s[0])
    horizon = int(np.searchsorted(t_s, t0 + bias_horizon_s))
    horizon = max(horizon, min(n, 2))
    good = usable[:horizon] & ~failed[:horizon]
    gy = np.asarray(gyro[:horizon], dtype=np.float64)
    ac = np.asarray(accel[:horizon], dtype=np.float64)
    a_all = np.linalg.norm(ac, axis=1)
    t_h = t_s[:horizon]
    n = horizon

    med_dt_h = float(np.median(np.diff(t_h))) if n > 1 else 0.005
    half = max(1, int(round(0.06 / max(med_dt_h, 1e-4))))
    g2 = gy * gy
    csum = np.vstack([np.zeros((1, 3)), np.cumsum(gy, axis=0)])
    csum2 = np.vstack([np.zeros((1, 3)), np.cumsum(g2, axis=0)])
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx + half + 1, 0, n)
    cnt = np.maximum(hi - lo, 1).astype(np.float64)[:, None]
    mean = (csum[hi] - csum[lo]) / cnt
    var = np.maximum((csum2[hi] - csum2[lo]) / cnt - mean * mean, 0.0)
    gstd = np.sqrt(var.sum(axis=1))
    noisy_imu = float(np.median(gstd[good])) > 0.015 if good.any() else False

    search = np.flatnonzero((t_h <= t0 + max(protect_s, 5.0)) & good)
    if search.size < 5:
        search = np.arange(min(200, n))
    a_n = a_all[search]
    w_n_raw = np.linalg.norm(gy[search], axis=1)
    score = np.abs(a_n - G) + 5.0 * w_n_raw
    k = int(min(40, search.size))
    best_local = np.argpartition(score, k - 1)[:k]
    best = search[best_local]
    static_init = (float(np.median(np.abs(a_n[best_local] - G))) < 0.5) and (
        float(np.median(w_n_raw[best_local])) < 0.08
    )

    near_g = good & (np.abs(a_all - G) < 0.22) & (a_all > 1.0)
    if noisy_imu and int(near_g.sum()) >= 80:
        cand = np.flatnonzero(near_g)
        g_med_all = np.median(gy[good], axis=0)
        resid = np.linalg.norm(gy[cand] - g_med_all, axis=1)
        score_b = resid + 3.0 * gstd[cand]
        kk = max(40, int(0.12 * cand.size))
        kk = min(kk, cand.size)
        pick = np.argpartition(score_b, kk - 1)[:kk]
        still = cand[pick]
        bg0 = np.median(gy[still], axis=0)
        am = np.mean(ac[still], axis=0)
    else:
        bg0 = np.median(gy[best], axis=0)
        am = np.mean(ac[best], axis=0)

    q0 = gravity_aligned_quat(float(am[0]), float(am[1]), float(am[2]))
    gbx, gby, gbz = _world_up_in_body(q0)
    ba0 = am - G * np.array([gbx, gby, gbz])
    return {
        "ba": ba0.astype(np.float64),
        "bg": np.asarray(bg0, dtype=np.float64),
        "q0": q0,
        "noisy_imu": bool(noisy_imu),
        "static_init": bool(static_init),
    }


def run_lean_ins(
    t_s: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    usable: np.ndarray,
    failed: np.ndarray,
    gap: np.ndarray,
    dup: np.ndarray,
    vel_hp_tau_s: float = 0.35,
    att_hz: float = 50.0,
    protect_s: float = 2.0,
    gravity_aid: bool = False,
    gravity_kp: float = 0.05,
):
    """50 Hz lean INS with tracemalloc-safe vectorized post-protect path.

    Block means + protect Mahony (short) + ``att_scan`` / ``hp_block`` (numpy).
    Keeps locked gate numerics while cutting traced ms/s under the FAQ 20 cap.
    """
    _ = gap
    n = len(t_s)
    init = estimate_biases_and_q0(t_s, gyro, accel, usable, failed, protect_s=protect_s)
    ba = init["ba"].astype(np.float64).copy()
    bg = init["bg"].astype(np.float64).copy()
    q = init["q0"].astype(np.float64).copy()
    _quat_normalize_inplace(q)

    t0 = float(t_s[0])
    if init["noisy_imu"]:
        horizon = int(np.searchsorted(t_s, t0 + 30.0))
        horizon = max(horizon, min(n, 2))
        med_dt = float(np.median(np.diff(t_s[:horizon]))) if horizon > 1 else 0.005
        half = max(1, int(round(0.25 / max(med_dt, 1e-4))))
        gy_h = np.asarray(gyro[:horizon], dtype=np.float64)
        ac_h = np.asarray(accel[:horizon], dtype=np.float64)
        a_var = _rolling_var_norm(ac_h, half)
        w_n = np.linalg.norm(gy_h - bg, axis=1)
        still = usable[:horizon] & ~failed[:horizon] & (w_n < 0.06) & (a_var < 0.04)
        if int(still.sum()) >= 40:
            bg = 0.85 * np.median(gy_h[still], axis=0) + 0.15 * bg
            am = np.mean(ac_h[still], axis=0)
            q_s = gravity_aligned_quat(float(am[0]), float(am[1]), float(am[2]))
            gbx, gby, gbz = _world_up_in_body(q_s)
            ba_m = am - G * np.array([gbx, gby, gbz])
            if np.linalg.norm(ba_m) < 0.5:
                ba = 0.75 * ba_m + 0.25 * ba
        del gy_h, ac_h, a_var, w_n

    med_dt = float(np.median(np.diff(t_s))) if n > 1 else 0.005
    step = max(1, int(round(1.0 / max(att_hz * max(med_dt, 1e-4), 1e-6))))
    idx = np.arange(0, n, step, dtype=np.int64)
    if len(idx) == 0 or idx[-1] != n - 1:
        idx = np.append(idx, n - 1) if n else idx
    m = len(idx)

    # Float32 cumsum block means — avoids per-keyframe memmap.mean under tracemalloc.
    good = usable & ~failed
    w32 = good.astype(np.float32)
    w_cs = np.zeros(n + 1, dtype=np.float32)
    np.cumsum(w32, out=w_cs[1:])
    n_cs = np.arange(n + 1, dtype=np.float32)

    def _pref_means(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pref_m = np.zeros((n + 1, 3), dtype=np.float32)
        pref_a = np.zeros((n + 1, 3), dtype=np.float32)
        for c in range(3):
            col = np.asarray(x[:, c], dtype=np.float32)
            np.cumsum(col, out=pref_a[1:, c])
            np.cumsum(col * w32, out=pref_m[1:, c])
            del col
        return pref_m, pref_a

    g_cs, g_all = _pref_means(gyro)
    a_cs, a_all = _pref_means(accel)
    del w32

    i0s = idx[:-1]
    i1s = idx[1:]
    lo = i0s + 1
    hi = i1s + 1
    wc = w_cs[hi] - w_cs[lo]
    use_mask = wc > 0.0
    g_mean = np.empty((m - 1, 3), dtype=np.float64)
    a_mean = np.empty((m - 1, 3), dtype=np.float64)
    if np.any(use_mask):
        g_mean[use_mask] = (g_cs[hi] - g_cs[lo])[use_mask] / wc[use_mask, None]
        a_mean[use_mask] = (a_cs[hi] - a_cs[lo])[use_mask] / wc[use_mask, None]
    if np.any(~use_mask):
        cnt = np.maximum(n_cs[hi] - n_cs[lo], 1.0)
        g_mean[~use_mask] = (g_all[hi] - g_all[lo])[~use_mask] / cnt[~use_mask, None]
        a_mean[~use_mask] = (a_all[hi] - a_all[lo])[~use_mask] / cnt[~use_mask, None]
    dt_k = (t_s[i1s] - t_s[i0s]).astype(np.float64)
    failed_k = failed[i1s]
    del g_cs, a_cs, g_all, a_all, w_cs, n_cs

    n_prot = int(np.searchsorted(t_s[idx], t0 + protect_s, side="right"))
    n_prot = max(1, min(m, n_prot))

    quat_d = np.zeros((m, 4), dtype=np.float64)
    quat_d[0] = q
    grav_updates = 0
    bax, bay, baz = float(ba[0]), float(ba[1]), float(ba[2])
    bgx, bgy, bgz = float(bg[0]), float(bg[1]), float(bg[2])
    gk = float(gravity_kp)
    f_body = np.empty((m - 1, 3), dtype=np.float64)
    omega_aid = np.empty((m - 1, 3), dtype=np.float64)

    # Short sequential protect Mahony only (~2 s of keyframes).
    for k in range(1, n_prot):
        j = k - 1
        i0 = int(i0s[j])
        i1 = int(i1s[j])
        dt = float(dt_k[j])
        if i1 <= i0 or dt <= 0.0 or bool(failed_k[j]):
            omega_aid[j] = 0.0
            f_body[j] = 0.0
            quat_d[k] = q
            continue
        wx = float(g_mean[j, 0] - bgx)
        wy = float(g_mean[j, 1] - bgy)
        wz = float(g_mean[j, 2] - bgz)
        fx = float(a_mean[j, 0] - bax)
        fy = float(a_mean[j, 1] - bay)
        fz = float(a_mean[j, 2] - baz)
        fn = (fx * fx + fy * fy + fz * fz) ** 0.5
        if abs(fn - G) < 0.8 and fn > 1.0:
            inv = 1.0 / max(fn, 1e-9)
            fhx, fhy, fhz = fx * inv, fy * inv, fz * inv
            gbx_b, gby_b, gbz_b = _world_up_in_body(q)
            ex = gby_b * fhz - gbz_b * fhy
            ey = gbz_b * fhx - gbx_b * fhz
            ez = gbx_b * fhy - gby_b * fhx
            wx += 0.25 * ex
            wy += 0.25 * ey
            wz += 0.25 * ez
            bax += 2e-3 * (fx - fn * gbx_b)
            bay += 2e-3 * (fy - fn * gby_b)
            baz += 2e-3 * (fz - fn * gbz_b)
            bgx -= 1e-4 * ex
            bgy -= 1e-4 * ey
            bgz -= 1e-4 * ez
            grav_updates += 1
        omega_aid[j] = (wx, wy, wz)
        f_body[j] = (fx, fy, fz)
        _quat_mul_right_rotvec(q, wx * dt, wy * dt, wz * dt)
        quat_d[k] = q

    # Post-protect: frozen bias + vectorized attitude scan (tracemalloc-safe).
    if n_prot < m:
        j0 = n_prot - 1
        ba_v = np.array([bax, bay, baz], dtype=np.float64)
        bg_v = np.array([bgx, bgy, bgz], dtype=np.float64)
        f_body[j0:] = a_mean[j0:] - ba_v
        omega_aid[j0:] = g_mean[j0:] - bg_v
        bad_post = (dt_k[j0:] <= 0.0) | failed_k[j0:] | (i1s[j0:] <= i0s[j0:])
        if np.any(bad_post):
            omega_aid[j0:][bad_post] = 0.0
            f_body[j0:][bad_post] = 0.0
        if gravity_aid:
            for j in range(j0, m - 1):
                if bad_post[j - j0]:
                    continue
                fx, fy, fz = f_body[j]
                wx, wy, wz = omega_aid[j]
                fn = (fx * fx + fy * fy + fz * fz) ** 0.5
                wn = (wx * wx + wy * wy + wz * wz) ** 0.5
                if wn < 0.05 and fn > 1.0:
                    inv = 1.0 / max(fn, 1e-9)
                    fhx, fhy, fhz = fx * inv, fy * inv, fz * inv
                    gbx_b, gby_b, gbz_b = _world_up_in_body(quat_d[j])
                    ex = gby_b * fhz - gbz_b * fhy
                    ey = gbz_b * fhx - gbx_b * fhz
                    ez = gbx_b * fhy - gby_b * fhx
                    omega_aid[j, 0] = wx + gk * ex
                    omega_aid[j, 1] = wy + gk * ey
                    omega_aid[j, 2] = wz + gk * ez
                    grav_updates += 1
        q_post = att_scan(q, omega_aid[j0:], dt_k[j0:])
        quat_d[n_prot:] = q_post
        del q_post
    del g_mean, a_mean

    # Vectorized world accel + cascaded HP (constant-α via med dt).
    tau = max(float(vel_hp_tau_s), 1e-3)
    aw = rotate_body_to_world(quat_d[1:], f_body)
    aw[:, 2] -= G
    an = np.linalg.norm(aw, axis=1)
    hot = an > 25.0
    if np.any(hot):
        aw[hot] *= (25.0 / an[hot])[:, None]
    bad_step = (dt_k <= 0.0) | failed_k | (i1s <= i0s)
    if np.any(bad_step):
        aw[bad_step] = 0.0
    dt_safe = dt_k.copy()
    dt_safe[dt_safe <= 0.0] = med_dt
    v_raw = np.zeros((m, 3), dtype=np.float64)
    v_raw[1:] = np.cumsum(aw * dt_safe[:, None], axis=0)
    dt_full = np.concatenate([[med_dt], dt_safe])
    if np.any(failed_k):
        cut = int(np.flatnonzero(failed_k)[0]) + 1
        cut = max(1, min(m, cut))
        pos_good, _vhp = hp_block(v_raw[:cut], dt_full[:cut], tau)
        pos64 = np.empty((m, 3), dtype=np.float64)
        pos64[:cut] = pos_good
        pos64[cut:] = pos_good[-1]
        del pos_good, _vhp
    else:
        pos64, _vhp = hp_block(v_raw, dt_full, tau)
        del _vhp
    del aw, f_body, omega_aid, v_raw

    return {
        "t_d": t_s[idx],
        "pos_d": pos64.astype(np.float32),
        "quat_d": quat_d.astype(np.float32),
        "idx": idx,
        "ba": np.array([bax, bay, baz]),
        "bg": np.array([bgx, bgy, bgz]),
        "q0": init["q0"],
        "noisy_imu": init["noisy_imu"],
        "grav_updates": grav_updates,
        "decim_step": int(step),
        "n_keyframes": m,
        "full_rate": False,
    }
