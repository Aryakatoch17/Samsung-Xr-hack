#!/usr/bin/env python3
"""Generate a clean synthetic EuRoC-style IMU CSV for local smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

G = 9.80665


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/clean_imu_euroc.csv")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--rate", type=float, default=200.0)
    args = ap.parse_args()

    n = int(args.seconds * args.rate)
    dt = 1.0 / args.rate
    t0_ns = 1_403_636_579_758_555_392
    ts = t0_ns + (np.arange(n) * dt * 1e9).astype(np.int64)

    # Gentle curved trajectory in world (z-up)
    t = np.arange(n) * dt
    # world accel (excluding gravity) — mild centripetal-ish motion
    ax_w = 0.4 * np.sin(0.3 * t)
    ay_w = 0.3 * np.cos(0.25 * t)
    az_w = 0.05 * np.sin(0.5 * t)
    a_w = np.stack([ax_w, ay_w, az_w], axis=1)

    # constant level attitude with tiny yaw rate
    yaw_rate = 0.05
    gyro = np.zeros((n, 3))
    gyro[:, 2] = yaw_rate

    # body ≈ world for this simple test (identity attitude)
    # specific force f = a - g_w, g_w=[0,0,-G] => f = a + [0,0,G]
    accel = a_w + np.array([0.0, 0.0, G])

    # EuRoC column order: ts, gx,gy,gz, ax,ay,az — write ts as full integer ns
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
        "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for i in range(n):
            f.write(
                f"{int(ts[i])},{gyro[i,0]:.8f},{gyro[i,1]:.8f},{gyro[i,2]:.8f},"
                f"{accel[i,0]:.8f},{accel[i,1]:.8f},{accel[i,2]:.8f}\n"
            )
    print(f"wrote {path}  rows={n}  duration={args.seconds}s")


if __name__ == "__main__":
    main()
