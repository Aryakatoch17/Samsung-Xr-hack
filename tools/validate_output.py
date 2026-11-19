#!/usr/bin/env python3
"""Validate estimated_trajectory.txt against an IMU CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.io_utils import load_imu_csv, resolve_imu_path  # noqa: E402


def validate(imu_path: Path, traj_path: Path) -> list[str]:
    errs = []
    imu = load_imu_csv(imu_path)
    n = imu["n"]
    raw = np.loadtxt(traj_path, comments="#")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] != 8:
        errs.append(f"expected 8 columns, got {raw.shape[1]}")
        return errs
    if raw.shape[0] != n:
        errs.append(f"row count {raw.shape[0]} != input {n}")
    if not np.isfinite(raw).all():
        errs.append("non-finite values in trajectory")
    q = raw[:, 4:8]
    nrm = np.linalg.norm(q, axis=1)
    if np.max(np.abs(nrm - 1.0)) > 1e-3:
        errs.append(f"quaternions not unit (max |||q||-1|={np.max(np.abs(nrm-1)):.3e})")
    # Timestamps should match input (within 1 us after unit conversion)
    if abs(float(raw[0, 0] - imu["t_s"][0])) > 1e-3:
        errs.append("first timestamp mismatch vs IMU")
    if abs(float(raw[-1, 0] - imu["t_s"][-1])) > 1e-3:
        errs.append("last timestamp mismatch vs IMU")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("imu")
    ap.add_argument("traj")
    args = ap.parse_args()
    imu = resolve_imu_path(args.imu)
    errs = validate(imu, Path(args.traj))
    if errs:
        print("INVALID")
        for e in errs:
            print(" -", e)
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
