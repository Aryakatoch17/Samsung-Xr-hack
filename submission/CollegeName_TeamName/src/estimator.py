"""Full Track-3 trajectory estimation pipeline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .fast_ins import reintegrate_fast, run_pass1, smooth_bias
from .io_utils import load_imu_csv, write_tum_trajectory
from .quality import analyze_imu


def _estimate_peak_algorithm_bytes(arrays: list[np.ndarray]) -> int:
    total = sum(int(a.nbytes) for a in arrays if a is not None)
    total += 8192
    return total


def run_estimator(imu_csv: str, out_traj: str) -> dict[str, Any]:
    t_wall0 = time.perf_counter()
    imu = load_imu_csv(imu_csv)
    t_s = imu["t_s"]
    gyro = imu["gyro"]
    accel = imu["accel"]
    n = imu["n"]
    duration = float(t_s[-1] - t_s[0]) if n > 1 else 0.0

    qual = analyze_imu(t_s, accel, gyro)
    usable = qual["usable"]
    failed = qual["failed"]
    gap = qual["gap"]
    dup = qual["dup"]
    spike = qual["spike"]
    stuck = qual["stuck"]

    pass1 = run_pass1(t_s, gyro, accel, usable, failed, gap, dup)

    ba_s = smooth_bias(pass1["ba"], usable & ~failed, win=51)
    bg_s = smooth_bias(pass1["bg"], usable & ~failed, win=51)
    if failed.any():
        fs = qual["fail_start_idx"]
        if fs > 0:
            ba_s[fs:] = ba_s[fs - 1]
            bg_s[fs:] = bg_s[fs - 1]

    # Blind 5 s: constant-velocity coast (a=0). Noisy pre-fail accel hurts more than it helps
    # over a 5 s open-loop window (0.5*a*t^2). Keep a robust angular-rate prior.
    fs = qual["fail_start_idx"]
    coast_a = np.zeros(3)
    if fs < n and fs > 0:
        t_lo = t_s[fs] - 1.0
        mask = (t_s >= t_lo) & (t_s < t_s[fs]) & usable & ~failed
        if int(mask.sum()) >= 10:
            ww = pass1["omega"][mask]
            coast_w = np.median(ww, axis=0)
            # reject absurd spin
            if float(np.linalg.norm(coast_w)) > 1.5:
                coast_w = pass1["omega"][fs - 1].copy()
        else:
            coast_w = pass1["omega"][fs - 1].copy()
    else:
        coast_w = pass1["omega"][-1].copy()

    wn = float(np.linalg.norm(coast_w))
    if wn > 1.5:
        coast_w = coast_w * (1.5 / wn)

    pos, vel, quat = reintegrate_fast(
        t_s,
        gyro,
        accel,
        pass1["q0"],
        ba_s,
        bg_s,
        usable,
        failed,
        gap,
        dup,
        coast_a,
        coast_w,
    )

    write_tum_trajectory(out_traj, t_s, pos, quat)

    elapsed = time.perf_counter() - t_wall0
    peak_bytes = _estimate_peak_algorithm_bytes(
        [
            t_s,
            gyro,
            accel,
            pass1["ba"],
            pass1["bg"],
            pass1["quat"],
            pass1["a_w"],
            pass1["omega"],
            ba_s,
            bg_s,
            pos,
            vel,
            quat,
        ]
    )
    ms_per_s = (elapsed * 1000.0) / max(duration, 1e-6)

    return {
        "n_samples": n,
        "duration_s": duration,
        "runtime_s": elapsed,
        "ms_per_imu_second": ms_per_s,
        "peak_algorithm_bytes": peak_bytes,
        "peak_algorithm_mb": peak_bytes / (1024 * 1024),
        "column_order_detected": imu["order"],
        "ts_unit": imu["ts_unit"],
        "acc_unit": imu["acc_unit"],
        "fail_start_idx": qual["fail_start_idx"],
        "n_failed": int(failed.sum()),
        "n_spikes": int(spike.sum()),
        "n_stuck": int(stuck.sum()),
        "n_gaps": int(gap.sum()),
        "n_dups": int(dup.sum()),
        "gravity_updates": int(pass1["grav_updates"]),
        "coast_accel_world": coast_a.tolist(),
        "coast_omega": coast_w.tolist(),
        "output": out_traj,
    }
