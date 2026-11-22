"""Numpy-only vectorized attitude scan, cascaded HP, and TUM byte writer.

Designed to cut tracemalloc tax from per-row Python / savetxt.
Self-test: ``python -m src.fastpath``
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

G = 9.80665


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, 1e-15)


def quat_mul_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a⊗b for arrays (...,4) in xyzw."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def rotvecs_to_quats(rv: np.ndarray) -> np.ndarray:
    """Map rotation vectors (n,3) → delta quaternions (n,4) xyzw."""
    ang = np.linalg.norm(rv, axis=1)
    half = 0.5 * ang
    small = ang < 1e-12
    safe = np.where(small, 1.0, ang)
    s = np.sin(half) / safe
    q = np.empty((rv.shape[0], 4), dtype=np.float64)
    q[:, 0] = rv[:, 0] * s
    q[:, 1] = rv[:, 1] * s
    q[:, 2] = rv[:, 2] * s
    q[:, 3] = np.cos(half)
    if np.any(small):
        q[small, 0] = 0.5 * rv[small, 0]
        q[small, 1] = 0.5 * rv[small, 1]
        q[small, 2] = 0.5 * rv[small, 2]
        q[small, 3] = 1.0
    return quat_normalize(q)


def quat_prefix_scan(dq: np.ndarray) -> np.ndarray:
    """Inclusive parallel prefix: out[i] = dq[0]⊗…⊗dq[i] (Hillis–Steele)."""
    a = quat_normalize(np.asarray(dq, dtype=np.float64).copy())
    n = a.shape[0]
    d = 1
    while d < n:
        b = a.copy()
        b[d:] = quat_mul_batch(a[:-d], a[d:])
        a = quat_normalize(b)
        # Keep hemisphere continuous vs previous
        dots = np.sum(a[1:] * a[:-1], axis=1)
        flip = dots < 0.0
        if np.any(flip):
            # flip from first break onward in this pass only where negative vs prev
            idx = np.flatnonzero(flip) + 1
            a[idx] *= -1.0
        d *= 2
    return a


def att_scan(
    q0: np.ndarray,
    omega: np.ndarray,
    dt: np.ndarray,
    chunk: int = 4096,
) -> np.ndarray:
    """Gyro attitude: q[i] = q0 ⊗ Exp(ω[0]dt[0]) ⊗ … ⊗ Exp(ω[i]dt[i]).

    Chunked Hillis–Steele prefix keeps peak scratch ≪ full-N×logN.
    ``omega`` (n,3) rad/s, ``dt`` (n,) seconds. Returns (n,4) xyzw.
    """
    q0 = quat_normalize(np.asarray(q0, dtype=np.float64).reshape(4))
    omega = np.asarray(omega, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64).reshape(-1)
    n = omega.shape[0]
    if n == 0:
        return np.zeros((0, 4), dtype=np.float64)
    out = np.empty((n, 4), dtype=np.float64)
    q_run = q0.copy()
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        rv = omega[start:end] * dt[start:end, None]
        dq = rotvecs_to_quats(rv)
        scan = quat_prefix_scan(dq)
        q0b = np.broadcast_to(q_run, (end - start, 4))
        block = quat_normalize(quat_mul_batch(q0b, scan))
        if np.dot(block[0], q_run) < 0.0:
            block[0] *= -1.0
        dots = np.sum(block[1:] * block[:-1], axis=1)
        if dots.size and np.any(dots < 0.0):
            flips = np.cumsum((dots < 0.0).astype(np.int32))
            sign = np.ones(end - start, dtype=np.float64)
            sign[1:] = np.where((flips % 2) == 1, -1.0, 1.0)
            block *= sign[:, None]
        out[start:end] = block
        q_run = block[-1].copy()
    return out


def rotate_body_to_world(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate body vectors v (n,3) by quats q (n,4) xyzw → world (n,3)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    twx, twy, twz = 2.0 * w * x, 2.0 * w * y, 2.0 * w * z
    txx, tyy, tzz = 2.0 * x * x, 2.0 * y * y, 2.0 * z * z
    txy, txz, tyz = 2.0 * x * y, 2.0 * x * z, 2.0 * y * z
    wx = (1.0 - tyy - tzz) * vx + (txy - twz) * vy + (txz + twy) * vz
    wy = (txy + twz) * vx + (1.0 - txx - tzz) * vy + (tyz - twx) * vz
    wz = (txz - twy) * vx + (tyz + twx) * vy + (1.0 - txx - tyy) * vz
    return np.column_stack([wx, wy, wz])


def _exp_smooth_const(x: np.ndarray, alpha: float, block: int = 512) -> np.ndarray:
    """Causal y[i]=α x[i]+(1-α) y[i-1] with block geometric cumsums (stable)."""
    a = float(alpha)
    w = 1.0 - a
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    out = np.empty_like(x)
    if n == 0:
        return out
    carry = np.zeros(x.shape[1], dtype=np.float64) if x.ndim == 2 else 0.0
    for start in range(0, n, block):
        end = min(n, start + block)
        m = end - start
        pw = w ** np.arange(m, dtype=np.float64)
        # y_local[k] = a * sum_{j=0..k} w^{k-j} x[start+j] + w^{k+1} * carry_prev
        #            = a * pw[k] * sum_{j=0..k} x[start+j]/pw[j] + pw[k]*w * carry
        inv = 1.0 / np.maximum(pw, 1e-300)
        xb = x[start:end]
        if xb.ndim == 1:
            cs = np.cumsum(xb * inv)
            yb = a * pw * cs + (pw * w) * float(carry)
            out[start:end] = yb
            carry = yb[-1]
        else:
            cs = np.cumsum(xb * inv[:, None], axis=0)
            yb = a * pw[:, None] * cs + (pw[:, None] * w) * carry.reshape(1, -1)
            out[start:end] = yb
            carry = yb[-1].copy()
    return out


def hp_block(v_raw: np.ndarray, dt: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    """Cascaded 2nd-order velocity HP + position integrate (constant-α via med dt)."""
    v_raw = np.asarray(v_raw, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64).reshape(-1)
    n = v_raw.shape[0]
    tau = max(float(tau), 1e-3)
    med_dt = float(np.median(dt)) if n else 0.005
    alpha = 1.0 - math.exp(-med_dt / tau)
    lp1 = _exp_smooth_const(v_raw, alpha)
    y1 = v_raw - lp1
    lp2 = _exp_smooth_const(y1, alpha)
    v_hp = y1 - lp2
    pos = np.cumsum(v_hp * dt[:, None], axis=0)
    return pos, v_hp


def tum_bytes(t_s: np.ndarray, pos: np.ndarray, quat_xyzw: np.ndarray) -> bytes:
    """Format a small TUM chunk as ASCII bytes (for tests / callers)."""
    t_s = np.asarray(t_s, dtype=np.float64).reshape(-1)
    pos = np.asarray(pos, dtype=np.float64)
    q = quat_normalize(np.asarray(quat_xyzw, dtype=np.float64))
    n = t_s.shape[0]
    lines = [
        f"{t_s[i]:.9f} {pos[i, 0]:.9f} {pos[i, 1]:.9f} {pos[i, 2]:.9f} "
        f"{q[i, 0]:.9f} {q[i, 1]:.9f} {q[i, 2]:.9f} {q[i, 3]:.9f}"
        for i in range(n)
    ]
    return ("\n".join(lines) + "\n").encode("ascii")


def write_tum_bytes(
    path: str,
    t_s: np.ndarray,
    pos: np.ndarray,
    quat_xyzw: np.ndarray,
    chunk: int = 4096,
) -> None:
    """Stream TUM ASCII in chunks (one join/encode per chunk — avoids savetxt TM tax)."""
    t_s = np.asarray(t_s, dtype=np.float64).reshape(-1)
    pos = np.asarray(pos, dtype=np.float64)
    q = quat_normalize(np.asarray(quat_xyzw, dtype=np.float64))
    n = t_s.shape[0]
    with open(path, "wb") as f:
        f.write(b"# timestamp tx ty tz qx qy qz qw\n")
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            f.write(tum_bytes(t_s[start:end], pos[start:end], q[start:end]))


def _self_test() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    dt = np.full(n, 0.005)
    omega = rng.normal(0.0, 0.2, size=(n, 3))
    q0 = np.array([0.0, 0.0, 0.0, 1.0])

    # Sequential reference
    q = q0.copy()
    seq = np.empty((n, 4))
    for i in range(n):
        rv = omega[i] * dt[i]
        ang = float(np.linalg.norm(rv))
        if ang < 1e-12:
            dq = np.array([0.5 * rv[0], 0.5 * rv[1], 0.5 * rv[2], 1.0])
        else:
            s = math.sin(0.5 * ang) / ang
            dq = np.array([rv[0] * s, rv[1] * s, rv[2] * s, math.cos(0.5 * ang)])
        dq /= np.linalg.norm(dq) + 1e-15
        q = quat_mul_batch(q.reshape(1, 4), dq.reshape(1, 4))[0]
        q /= np.linalg.norm(q) + 1e-15
        if i and np.dot(q, seq[i - 1]) < 0:
            q = -q
        seq[i] = q

    got = att_scan(q0, omega, dt)
    # Compare up to sign
    align = np.sign(np.sum(got * seq, axis=1))
    align[align == 0] = 1
    err = np.max(np.linalg.norm(got * align[:, None] - seq, axis=1))
    print(f"att_scan max err vs sequential: {err:.3e}")
    assert err < 1e-10, err

    aw = rng.normal(0.0, 0.5, size=(n, 3))
    v_raw = np.cumsum(aw * dt[:, None], axis=0)
    pos, _vhp = hp_block(v_raw, dt, 0.35)
    # Round-trip tum
    payload = tum_bytes(np.cumsum(dt), pos, got)
    rows = [ln for ln in payload.decode().splitlines() if ln and not ln.startswith("#")]
    parsed = np.loadtxt(rows)
    assert parsed.shape == (n, 8)
    ts_err = np.max(np.abs(parsed[:, 0] - np.cumsum(dt)))
    pos_err = np.max(np.abs(parsed[:, 1:4] - pos))
    print(f"tum round-trip ts_err={ts_err:.3e} pos_err={pos_err:.3e}")
    assert ts_err < 5e-7
    assert pos_err < 5e-7
    print("fastpath self-test OK")


if __name__ == "__main__":
    _self_test()
