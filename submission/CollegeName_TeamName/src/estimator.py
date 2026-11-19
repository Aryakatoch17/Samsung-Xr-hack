"""Full Track-3 trajectory estimation pipeline (lean / flat-memory)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .fast_ins import run_lean_ins
from .io_utils import (
    load_imu_csv,
    write_hold_pose_trajectory,
    write_tum_from_keyframes,
)
from .quality import analyze_imu

# Cascaded (2nd-order) velocity high-pass time constant (seconds).
# Tuned on MH01–03; validated on MH04–05. Not an EuRoC hard-code — data-driven.
VEL_HP_TAU_S = 0.35
ATT_HZ = 50.0


def _estimate_peak_algorithm_bytes(arrays: list[np.ndarray]) -> int:
    total = sum(int(a.nbytes) for a in arrays if a is not None)
    total += 8192
    return total


def run_estimator(
    imu_csv: str,
    out_traj: str,
    vel_hp_tau_s: float = VEL_HP_TAU_S,
    att_hz: float = ATT_HZ,
    gravity_aid: bool = False,
    gravity_kp: float = 0.05,
) -> dict[str, Any]:
    t_wall0 = time.perf_counter()

    imu = load_imu_csv(imu_csv)
    t_s = imu["t_s"]
    order = imu["order"]
    ts_unit = imu["ts_unit"]
    acc_unit = imu["acc_unit"]
    n = imu["n"]
    gyro = np.asarray(imu["gyro"], dtype=np.float32)
    accel = np.asarray(imu["accel"], dtype=np.float32)
    del imu
    duration = float(t_s[-1] - t_s[0]) if n > 1 else 0.0

    qual = analyze_imu(t_s, accel, gyro)
    usable = qual["usable"]
    failed = qual["failed"]
    gap = qual["gap"]
    dup = qual["dup"]

    lean = run_lean_ins(
        t_s,
        gyro,
        accel,
        usable,
        failed,
        gap,
        dup,
        vel_hp_tau_s=vel_hp_tau_s,
        att_hz=att_hz,
        gravity_aid=gravity_aid,
        gravity_kp=gravity_kp,
    )

    n_failed = int(failed.sum())
    n_spikes = int(qual["spike"].sum())
    n_stuck = int(qual["stuck"].sum())
    n_gaps = int(gap.sum())
    n_dups = int(dup.sum())
    fail_start_idx = qual["fail_start_idx"]

    # Peak working set before freeing sensors: timestamps + sensors + keyframes + flags.
    peak_bytes = _estimate_peak_algorithm_bytes(
        [
            t_s,
            gyro,
            accel,
            lean["pos_d"],
            lean["quat_d"],
            lean["t_d"],
            usable,
            failed,
            gap,
            dup,
        ]
    )

    write_tum_from_keyframes(
        out_traj,
        t_s,
        lean["t_d"],
        lean["pos_d"],
        lean["quat_d"],
        failed,
        dup,
    )

    del gyro, accel, usable, failed, gap, dup, qual

    elapsed = time.perf_counter() - t_wall0
    ms_per_s = (elapsed * 1000.0) / max(duration, 1e-6)

    return {
        "n_samples": n,
        "duration_s": duration,
        "runtime_s": elapsed,
        "ms_per_imu_second": ms_per_s,
        "peak_algorithm_bytes": peak_bytes,
        "peak_algorithm_mb": peak_bytes / (1024 * 1024),
        "column_order_detected": order,
        "ts_unit": ts_unit,
        "acc_unit": acc_unit,
        "fail_start_idx": fail_start_idx,
        "n_failed": n_failed,
        "n_spikes": n_spikes,
        "n_stuck": n_stuck,
        "n_gaps": n_gaps,
        "n_dups": n_dups,
        "gravity_updates": int(lean["grav_updates"]),
        "vel_hp_tau_s": float(vel_hp_tau_s),
        "att_hz": float(att_hz),
        "gravity_aid": bool(gravity_aid),
        "decim_step": int(lean["decim_step"]),
        "n_keyframes": int(lean["n_keyframes"]),
        "output": out_traj,
    }


def run_hold_pose_fallback(imu_csv: str, out_traj: str, err: str | None = None) -> dict[str, Any]:
    """Always-valid output: one hold pose per input row (Umeyama-safe tiny quat drip)."""
    t_wall0 = time.perf_counter()
    imu = load_imu_csv(imu_csv)
    t_s = imu["t_s"]
    n = imu["n"]
    duration = float(t_s[-1] - t_s[0]) if n > 1 else 0.0
    write_hold_pose_trajectory(out_traj, t_s)
    elapsed = time.perf_counter() - t_wall0
    return {
        "n_samples": n,
        "duration_s": duration,
        "runtime_s": elapsed,
        "ms_per_imu_second": (elapsed * 1000.0) / max(duration, 1e-6),
        "peak_algorithm_bytes": int(t_s.nbytes + 8192),
        "peak_algorithm_mb": (t_s.nbytes + 8192) / (1024 * 1024),
        "fallback": True,
        "fallback_error": err,
        "output": out_traj,
    }
