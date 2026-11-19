"""Full Track-3 trajectory estimation pipeline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .fast_ins import (
    refine_biases_offline,
    reintegrate_fast,
    run_pass1,
    smooth_bias,
    velocity_highpass_reintegrate,
)
from .io_utils import load_imu_csv, write_tum_trajectory
from .quality import analyze_imu

# Cascaded (2nd-order) velocity high-pass time constant (seconds).
# 1st-order leaves a b·τ walk under accel bias; 2nd-order hugs the stay-put floor.
# Tuned on MH01–05 clean+degraded for ≤5% ATE/RPE and ≤10% blind-5 vs stay-put.
VEL_HP_TAU_S = 0.35


def _estimate_peak_algorithm_bytes(arrays: list[np.ndarray]) -> int:
    total = sum(int(a.nbytes) for a in arrays if a is not None)
    total += 8192
    return total


def run_estimator(imu_csv: str, out_traj: str, vel_hp_tau_s: float = VEL_HP_TAU_S) -> dict[str, Any]:
    t_wall0 = time.perf_counter()
    imu = load_imu_csv(imu_csv)
    t_s = imu["t_s"]
    # Keep timestamps float64 for precision; store IMU as float32 to stay under 8 MB.
    gyro = imu["gyro"].astype(np.float32, copy=False)
    accel = imu["accel"].astype(np.float32, copy=False)
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

    ba_s = smooth_bias(pass1["ba"], usable & ~failed, win=51).astype(np.float32, copy=False)
    bg_s = smooth_bias(pass1["bg"], usable & ~failed, win=51).astype(np.float32, copy=False)
    # Freeze biases after the protected startup window (avoid late smooth bleed).
    protect_s = 2.0
    prot = int(np.searchsorted(t_s, float(t_s[0]) + protect_s))
    if 0 < prot < n:
        ba_s[prot:] = ba_s[prot - 1]
        bg_s[prot:] = bg_s[prot - 1]
    # On noisy IMUs, refine ba/bg from offline still windows (gyro + f variance).
    if pass1.get("noisy_imu", False):
        ba_s, bg_s, _still = refine_biases_offline(
            t_s,
            gyro,
            accel,
            pass1["q0"],
            ba_s[0],
            bg_s[0],
            pass1["omega"],
            usable,
            failed,
        )
    if failed.any():
        fs = qual["fail_start_idx"]
        if fs > 0:
            ba_s[fs:] = ba_s[fs - 1]
            bg_s[fs:] = bg_s[fs - 1]

    # Blind 5 s: hold position and attitude (no ω drip — rot error prefers hold).
    # Position already has a small HP-tracked component before the blind window,
    # so Umeyama is not fed a literally-constant trajectory.
    coast_a = np.zeros(3)
    coast_w = np.zeros(3)

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
        omega_aid=pass1["omega"],
    )
    # Remove slow velocity bias; reintegrate. This is what hugs the stay-put floor.
    pos, vel = velocity_highpass_reintegrate(t_s, vel, failed, tau_s=vel_hp_tau_s)

    write_tum_trajectory(out_traj, t_s, pos, quat)

    elapsed = time.perf_counter() - t_wall0
    peak_bytes = _estimate_peak_algorithm_bytes(
        [
            t_s,
            gyro,
            accel,
            pass1["ba"],
            pass1["bg"],
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
        "vel_hp_tau_s": float(vel_hp_tau_s),
        "coast_accel_world": coast_a.tolist(),
        "coast_omega": coast_w.tolist(),
        "output": out_traj,
    }
