#!/usr/bin/env python3
"""Replay EuRoC GT into a kinematically consistent IMU (+ optional ADIS-like noise).

Produces a harder local benchmark than the smooth synthetic: real motion, real
rates, controllable sensor noise — without needing extra sequence downloads.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

G = 9.80665


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)


def rotmat_from_quat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-csv", default="data/euroc/MH_01_easy/gt.csv")
    ap.add_argument("--imu-csv", default="data/euroc/MH_01_easy/imu.csv")
    ap.add_argument("--out-dir", default="data/euroc_replay")
    ap.add_argument("--noise", action="store_true", help="Add ADIS16448-like noise/bias")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    gt = np.loadtxt(args.gt_csv, delimiter=",")
    # gt: ts, p, q(wxyz), v, bw, ba
    t_gt = gt[:, 0] * 1e-9
    p = gt[:, 1:4]
    # EuRoC quat is w,x,y,z → store x,y,z,w
    q = quat_normalize(np.column_stack([gt[:, 5], gt[:, 6], gt[:, 7], gt[:, 4]]))

    # Resample GT pose onto IMU timestamps for a fair replay
    imu = np.loadtxt(args.imu_csv, delimiter=",")
    t = imu[:, 0] * 1e-9
    # Interpolate position (linear) and attitude (slerp-lite via normalize lerp)
    j = np.searchsorted(t_gt, t, side="right") - 1
    j = np.clip(j, 0, len(t_gt) - 2)
    a = (t - t_gt[j]) / np.maximum(t_gt[j + 1] - t_gt[j], 1e-9)
    a = np.clip(a, 0.0, 1.0)[:, None]
    p_i = (1 - a) * p[j] + a * p[j + 1]
    q_i = quat_normalize((1 - a) * q[j] + a * q[j + 1])

    dt = np.diff(t, prepend=t[0] + (t[1] - t[0]))
    dt[0] = dt[1]
    # Body rate from quaternion difference: 2 * q^{-1} ⊗ q_dot
    gyro = np.zeros((len(t), 3))
    accel = np.zeros((len(t), 3))
    for i in range(1, len(t)):
        dq = quat_mul(quat_conj(q_i[i - 1]), q_i[i])
        if dq[3] < 0:
            dq = -dq
        # small-angle: [x,y,z] ≈ 0.5 * omega * dt
        w = 2.0 * dq[:3] / max(float(dt[i]), 1e-6)
        gyro[i] = w
        R = rotmat_from_quat(q_i[i])
        # world accel from finite-diff velocity
    vel = np.gradient(p_i, t, axis=0)
    acc_w = np.gradient(vel, t, axis=0)
    for i in range(len(t)):
        R = rotmat_from_quat(q_i[i])
        # specific force in body: R^T (a_w - g_w) with g_w=[0,0,-G]
        accel[i] = R.T @ (acc_w[i] - np.array([0.0, 0.0, -G]))

    gyro[0] = gyro[1]
    bg = np.array([-0.0028, 0.0210, 0.0796])
    ba = np.array([-0.03, 0.002, 0.01])
    if args.noise:
        # Rough ADIS16448-ish densities
        gyro = gyro + bg + rng.normal(0.0, 0.01, gyro.shape)
        accel = accel + ba + rng.normal(0.0, 0.08, accel.shape)
        tag = "noisy"
    else:
        gyro = gyro + bg
        accel = accel + ba
        tag = "clean"

    # EuRoC IMU CSV: ts, gx,gy,gz, ax,ay,az
    ts_ns = (t * 1e9).astype(np.int64)
    imu_out = np.column_stack([ts_ns, gyro, accel])
    imu_path = out / f"imu_{tag}.csv"
    np.savetxt(imu_path, imu_out, delimiter=",", fmt=["%d"] + ["%.10f"] * 6)

    # TUM GT on same stamps
    tum = np.column_stack([t, p_i, q_i])
    gt_path = out / "gt_tum.txt"
    np.savetxt(gt_path, tum, fmt="%.9f")
    print(f"wrote {imu_path} rows={len(t)} noise={args.noise}")
    print(f"wrote {gt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
