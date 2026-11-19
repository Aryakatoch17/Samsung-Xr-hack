"""Quaternion and rotation helpers (float32-friendly)."""

from __future__ import annotations

import numpy as np

G = 9.80665


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=v.dtype,
    )


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=q.dtype)
    return q / n


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product. q = [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=q1.dtype,
    )


def quat_from_rotvec(rv: np.ndarray) -> np.ndarray:
    """Small/finite rotation vector -> quaternion [x,y,z,w]."""
    angle = np.linalg.norm(rv)
    if angle < 1e-12:
        return np.array([0.5 * rv[0], 0.5 * rv[1], 0.5 * rv[2], 1.0], dtype=rv.dtype)
    axis = rv / angle
    s = np.sin(0.5 * angle)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(0.5 * angle)], dtype=rv.dtype)


def rotmat_from_quat(q: np.ndarray) -> np.ndarray:
    """Body -> world rotation matrix from quaternion [x,y,z,w]."""
    x, y, z, w = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=q.dtype,
    )


def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation that maps unit vector a to unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = np.cross(np.array([1.0, 0.0, 0.0], dtype=a.dtype), a)
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(np.array([0.0, 1.0, 0.0], dtype=a.dtype), a)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return quat_from_rotvec(np.pi * axis)
    q = np.array([v[0], v[1], v[2], 1.0 + c], dtype=a.dtype)
    return quat_normalize(q)


def gravity_aligned_quat(accel_mean: np.ndarray) -> np.ndarray:
    """
    Initialize orientation so measured accel (includes +g in body when static
    for EuRoC convention a = R^T (a_world - g)?).

    EuRoC IMU reports specific force in sensor frame. At rest, ||a|| ≈ g and
    the measured accel points opposite to gravity in the body if gravity is
    +9.81 in world-z (ASL/EuRoC: gravity is along +z of world? Actually EuRoC
    world has z-up and accelerometer measures specific force f = a - g with
    g = [0,0,-9.81], so at rest f ≈ -g = [0,0,+9.81]).

    We treat 'up' in world as +Z, so rest accel ≈ [0,0,+G] in world.
    Map measured body accel direction to world +Z.
    """
    a = accel_mean.astype(np.float64)
    n = np.linalg.norm(a)
    if n < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    a_body = a / n
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # R maps body -> world; we want R @ a_body = up
    return quat_from_two_vectors(a_body, up)
