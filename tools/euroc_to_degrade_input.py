#!/usr/bin/env python3
"""Convert EuRoC IMU CSV (ts,gyro,accel) to degrade_imu.py order (ts,accel,gyro)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import load_imu_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    args = ap.parse_args()

    imu = load_imu_csv(args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("timestamp,ax,ay,az,gx,gy,gz\n")
        for i in range(imu["n"]):
            if imu["ts_unit"] == "ns":
                ts_str = str(int(imu["t_ns"][i]))
            else:
                ts_str = f"{imu['t_s'][i]:.9f}"
            a = imu["accel"][i]
            g = imu["gyro"][i]
            f.write(
                f"{ts_str},{a[0]:.8f},{a[1]:.8f},{a[2]:.8f},"
                f"{g[0]:.8f},{g[1]:.8f},{g[2]:.8f}\n"
            )
    print(f"{args.input} -> {args.output}  order=accel_gyro  rows={imu['n']}")


if __name__ == "__main__":
    main()
