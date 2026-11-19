"""Allocation-light complementary INS (pass 1 + reintegrate)."""

from __future__ import annotations

import numpy as np

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
    """R^T @ [0,0,1]."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    # third row of R (body<-world for up), actually R.T @ ez = third column of R
    # R[:,2] =
    twx, twy = 2.0 * w * x, 2.0 * w * y
    txx, tyy = 2.0 * x * x, 2.0 * y * y
    txz, tyz = 2.0 * x * z, 2.0 * y * z
    # column 2:
    return (txz + twy), (tyz - twx), (1.0 - txx - tyy)


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
    ba = np.zeros((n, 3))
    bg = np.zeros((n, 3))
    quat = np.zeros((n, 4))
    a_w = np.zeros((n, 3))
    omega_h = np.zeros((n, 3))

    t0 = float(t_s[0])
    T = max(float(t_s[-1] - t0), 1e-6)
    prot = (t_s <= t0 + protect_s) & usable & ~failed
    init_idx = np.flatnonzero(prot)
    if init_idx.size < 5:
        init_idx = np.arange(min(50, n))

    # Gyro bias only if the protect window is quasi-static.
    # Otherwise subtracting mean(gyro) removes real angular rate and dumps gravity into accel.
    g_init = gyro[init_idx]
    g_norm_med = float(np.median(np.linalg.norm(g_init, axis=1)))
    if g_norm_med < 0.05:
        bgx = float(np.mean(g_init[:, 0]))
        bgy = float(np.mean(g_init[:, 1]))
        bgz = float(np.mean(g_init[:, 2]))
    else:
        bgx = bgy = bgz = 0.0

    bax = bay = baz = 0.0
    am = np.mean(accel[init_idx], axis=0)
    q = gravity_aligned_quat(float(am[0]), float(am[1]), float(am[2]))
    px = py = pz = 0.0
    vx = vy = vz = 0.0
    last_ax = last_ay = last_az = 0.0
    last_wx = last_wy = last_wz = 0.0
    grav_updates = 0

    gx_w, gy_w, gz_w = 0.0, 0.0, -G

    for i in range(n):
        if i == 0:
            ba[0] = (bax, bay, baz)
            bg[0] = (bgx, bgy, bgz)
            quat[0] = q
            continue

        dt = float(t_s[i] - t_s[i - 1])
        if dup[i] or dt <= 0.0:
            ba[i] = (bax, bay, baz)
            bg[i] = (bgx, bgy, bgz)
            quat[i] = q
            a_w[i] = (last_ax, last_ay, last_az)
            omega_h[i] = (last_wx, last_wy, last_wz)
            continue

        if failed[i] or ((gap[i] and dt > 0.05) or (not usable[i])):
            _quat_mul_right_rotvec(q, last_wx * dt, last_wy * dt, last_wz * dt)
            ax_, ay_, az_ = last_ax, last_ay, last_az
            px += vx * dt + 0.5 * ax_ * dt * dt
            py += vy * dt + 0.5 * ay_ * dt * dt
            pz += vz * dt + 0.5 * az_ * dt * dt
            vx += ax_ * dt
            vy += ay_ * dt
            vz += az_ * dt
            ba[i] = (bax, bay, baz)
            bg[i] = (bgx, bgy, bgz)
            quat[i] = q
            a_w[i] = (last_ax, last_ay, last_az)
            omega_h[i] = (last_wx, last_wy, last_wz)
            continue

        wx = float(gyro[i, 0]) - bgx
        wy = float(gyro[i, 1]) - bgy
        wz = float(gyro[i, 2]) - bgz
        fx = float(accel[i, 0]) - bax
        fy = float(accel[i, 1]) - bay
        fz = float(accel[i, 2]) - baz
        fn = (fx * fx + fy * fy + fz * fz) ** 0.5
        wn = (wx * wx + wy * wy + wz * wz) ** 0.5

        t_frac = (float(t_s[i]) - t0) / T
        sev = _sev(t_frac)

        # Gravity / bias aiding ONLY in the opening protect window.
        in_protect = (float(t_s[i]) - t0) <= protect_s
        if in_protect and (abs(fn - G) < 1.0) and (fn > 1.0):
            inv = 1.0 / fn
            fhx, fhy, fhz = fx * inv, fy * inv, fz * inv
            gbx, gby, gbz = _world_up_in_body(q)
            ex = gby * fhz - gbz * fhy
            ey = gbz * fhx - gbx * fhz
            ez = gbx * fhy - gby * fhx
            kp = 0.15 / (0.5 + sev)
            wx += kp * ex
            wy += kp * ey
            wz += kp * ez
            ba_a = 1e-4 / (0.5 + sev)
            bg_a = 5e-5 / (0.5 + sev)
            bax += ba_a * (fx - fn * gbx)
            bay += ba_a * (fy - fn * gby)
            baz += ba_a * (fz - fn * gbz)
            bgx -= bg_a * ex
            bgy -= bg_a * ey
            bgz -= bg_a * ez
            grav_updates += 1

        _quat_mul_right_rotvec(q, wx * dt, wy * dt, wz * dt)
        awx, awy, awz = _rot_body_to_world(q, fx, fy, fz)
        awx += gx_w
        awy += gy_w
        awz += gz_w
        an = (awx * awx + awy * awy + awz * awz) ** 0.5
        if an > 30.0:
            s = 30.0 / an
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

        ba[i] = (bax, bay, baz)
        bg[i] = (bgx, bgy, bgz)
        quat[i] = q
        a_w[i] = (awx, awy, awz)
        omega_h[i] = (wx, wy, wz)

    return {
        "ba": ba,
        "bg": bg,
        "quat": quat,
        "a_w": a_w,
        "omega": omega_h,
        "grav_updates": grav_updates,
        "q0": quat[0].copy(),
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
):
    n = len(t_s)
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    quat = np.zeros((n, 4))
    q = q0.astype(np.float64).copy()
    _quat_normalize_inplace(q)
    px = py = pz = 0.0
    vx = vy = vz = 0.0
    last_ax, last_ay, last_az = float(coast_a[0]), float(coast_a[1]), float(coast_a[2])
    last_wx, last_wy, last_wz = float(coast_w[0]), float(coast_w[1]), float(coast_w[2])
    cax, cay, caz = last_ax, last_ay, last_az
    cwx, cwy, cwz = last_wx, last_wy, last_wz
    gx_w, gy_w, gz_w = 0.0, 0.0, -G

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
            _quat_mul_right_rotvec(q, cwx * dt, cwy * dt, cwz * dt)
            px += vx * dt + 0.5 * cax * dt * dt
            py += vy * dt + 0.5 * cay * dt * dt
            pz += vz * dt + 0.5 * caz * dt * dt
            vx += cax * dt
            vy += cay * dt
            vz += caz * dt
            quat[i] = q
            pos[i] = (px, py, pz)
            vel[i] = (vx, vy, vz)
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

        wx = float(gyro[i, 0]) - float(bg[i, 0])
        wy = float(gyro[i, 1]) - float(bg[i, 1])
        wz = float(gyro[i, 2]) - float(bg[i, 2])
        fx = float(accel[i, 0]) - float(ba[i, 0])
        fy = float(accel[i, 1]) - float(ba[i, 1])
        fz = float(accel[i, 2]) - float(ba[i, 2])
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
