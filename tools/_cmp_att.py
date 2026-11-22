#!/usr/bin/env python3
"""Compare att_scan vs sequential on V1 keyframe omegas."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from degrade_imu import CONFIG, degrade
from src.fast_ins import (
    _quat_mul_right_rotvec,
    _quat_normalize_inplace,
    estimate_biases_and_q0,
    run_lean_ins,
)
from src.fastpath import att_scan
from src.io_utils import load_imu_csv
from src.quality import analyze_imu


def main() -> None:
    imu_path = ROOT / "outputs" / "holdout_verify" / "V1_01_easy" / "imu_degraded.csv"
    if not imu_path.exists():
        df = pd.read_csv(ROOT / "data" / "euroc" / "V1_01_easy" / "imu.csv")
        deg, _ = degrade(df, deepcopy(CONFIG), 42)
        imu_path.parent.mkdir(parents=True, exist_ok=True)
        deg.to_csv(imu_path, index=False)
    imu = load_imu_csv(str(imu_path))
    q = analyze_imu(imu["t_s"], imu["accel"], imu["gyro"])
    lean = run_lean_ins(
        imu["t_s"], imu["gyro"], imu["accel"], q["usable"], q["failed"], q["gap"], q["dup"]
    )
    # Rebuild means-like omega from lean biases and re-integrate sequential vs scan
    # Use lean quat as reference; also force a pure gyro scan from q0
    init = estimate_biases_and_q0(
        imu["t_s"], imu["gyro"], imu["accel"], q["usable"], q["failed"]
    )
    print("noisy", init["noisy_imu"], "keys", lean["n_keyframes"], "step", lean["decim_step"])
    # angle between consecutive lean quats
    qd = lean["quat_d"].astype(np.float64)
    dots = np.abs(np.sum(qd[1:] * qd[:-1], axis=1))
    ang = 2 * np.arccos(np.clip(dots, 0, 1)) * 180 / np.pi
    print("lean consecutive quat jump deg: max", ang.max(), "p99", np.percentile(ang, 99))


if __name__ == "__main__":
    main()
