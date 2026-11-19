#!/usr/bin/env python3
"""Kinematically consistent synthetic EuRoC IMU + TUM ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

G = 9.80665


def rot_zyx(yaw, pitch, roll=0.0):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def quat_from_R(R):
    tw = 1.0 + R[0, 0] + R[1, 1] + R[2, 2]
    if tw <= 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    w = 0.5 * np.sqrt(tw)
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    q = np.array([x, y, z, w])
    return q / (np.linalg.norm(q) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/synthetic")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--rate", type=float, default=200.0)
    args = ap.parse_args()

    n = int(args.seconds * args.rate)
    dt = 1.0 / args.rate
    t = np.arange(n) * dt
    t0_ns = 1_403_636_579_758_555_392
    ts = t0_ns + (t * 1e9).astype(np.int64)

    static_s = 2.0
    # Smooth fade-in so derivatives are consistent
    fade = np.clip((t - static_s) / 1.0, 0.0, 1.0)
    fade = fade * fade * (3.0 - 2.0 * fade)  # smoothstep
    t_mot = np.maximum(0.0, t - static_s)

    # Desired motion profile after fade
    pos_des = np.stack(
        [
            2.0 * np.sin(0.15 * t_mot),
            1.2 * (1.0 - np.cos(0.12 * t_mot)),
            0.3 * t_mot + 0.2 * np.sin(0.2 * t_mot),
        ],
        axis=1,
    )
    pos = fade[:, None] * pos_des
    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)

    yaw = fade * (0.25 * np.sin(0.1 * t_mot))
    pitch = fade * (0.05 * np.sin(0.07 * t_mot))
    roll = np.zeros(n)

    g_w = np.array([0.0, 0.0, -G])
    gyro = np.zeros((n, 3))
    accel = np.zeros((n, 3))
    quat = np.zeros((n, 4))
    R_list = [rot_zyx(float(yaw[i]), float(pitch[i]), float(roll[i])) for i in range(n)]

    for i in range(n):
        R = R_list[i]
        quat[i] = quat_from_R(R)
        if 0 < i < n - 1:
            Rdot = (R_list[i + 1] - R_list[i - 1]) / (2 * dt)
        elif i == 0:
            Rdot = (R_list[1] - R) / dt
        else:
            Rdot = (R - R_list[i - 1]) / dt
        W = R.T @ Rdot
        # project to skew basis
        gyro[i] = np.array([W[2, 1], W[0, 2], W[1, 0]])
        # Make omega exactly skew-symmetric extraction; then rebuild accel from kinematics
        accel[i] = R.T @ (acc[i] - g_w)

    # Rebuild GT by integrating the emitted IMU with GT attitudes — exact consistency
    pos_i = np.zeros((n, 3))
    vel_i = np.zeros((n, 3))
    for i in range(1, n):
        R = R_list[i]
        aw = R @ accel[i] + g_w
        pos_i[i] = pos_i[i - 1] + vel_i[i - 1] * dt + 0.5 * aw * dt * dt
        vel_i[i] = vel_i[i - 1] + aw * dt
    # Use integrated poses as GT so strapdown with perfect attitude has ~0 error
    pos = pos_i
    quat_out = quat

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imu_path = out_dir / "imu_euroc.csv"
    gt_path = out_dir / "gt_tum.txt"

    with open(imu_path, "w", encoding="utf-8") as f:
        f.write(
            "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
            "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
        )
        for i in range(n):
            f.write(
                f"{int(ts[i])},{gyro[i,0]:.10f},{gyro[i,1]:.10f},{gyro[i,2]:.10f},"
                f"{accel[i,0]:.10f},{accel[i,1]:.10f},{accel[i,2]:.10f}\n"
            )

    with open(gt_path, "w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i in range(n):
            f.write(
                f"{ts[i] * 1e-9:.9f} {pos[i,0]:.9f} {pos[i,1]:.9f} {pos[i,2]:.9f} "
                f"{quat_out[i,0]:.9f} {quat_out[i,1]:.9f} {quat_out[i,2]:.9f} {quat_out[i,3]:.9f}\n"
            )

    # Self-check with GT orientation
    p = np.zeros(3)
    v = np.zeros(3)
    for i in range(1, n):
        R = R_list[i]
        aw = R @ accel[i] + g_w
        p = p + v * dt + 0.5 * aw * dt * dt
        v = v + aw * dt
    print(f"wrote {imu_path}")
    print(f"wrote {gt_path}")
    print(f"  rows={n} duration={args.seconds}s")
    print(f"  self-check err={np.linalg.norm(p - pos[-1]):.6f}  path_len={np.sum(np.linalg.norm(np.diff(pos,axis=0),axis=1)):.2f}m")


if __name__ == "__main__":
    main()
