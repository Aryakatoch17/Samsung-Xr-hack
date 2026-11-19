"""Sample quality flags: spikes, stuck, duplicates, gaps, sensor failure."""

from __future__ import annotations

import numpy as np


def analyze_imu(
    t_s: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    protect_seconds: float = 2.0,
    failure_norm_thresh: float = 0.15,
    failure_min_seconds: float = 1.0,
) -> dict:
    n = len(t_s)
    dt = np.empty(n, dtype=np.float64)
    dt[0] = np.median(np.diff(t_s)) if n > 1 else 0.005
    dt[1:] = np.diff(t_s)

    med_dt = float(np.nanmedian(dt[1:])) if n > 1 else 0.005
    if not np.isfinite(med_dt) or med_dt <= 0:
        med_dt = 0.005
    med_dt = max(med_dt, 1e-4)

    # Duplicates / non-positive dt
    dup = dt <= 1e-9
    dup[0] = False

    # Gaps: missing rows -> large dt jumps (script bursts are 5-50 samples)
    gap = dt > (3.5 * med_dt)
    gap[0] = False

    # Stuck: short freezes (degrade_imu stuck_len is 2..10). Long identical
    # stretches are real static segments, NOT sensor faults.
    stuck = np.zeros(n, dtype=bool)
    if n > 1:
        same = np.all(np.abs(accel[1:] - accel[:-1]) < 1e-9, axis=1) & np.all(
            np.abs(gyro[1:] - gyro[:-1]) < 1e-9, axis=1
        )
        # mark runs of identical samples with length in [2, 15]
        i = 0
        while i < len(same):
            if not same[i]:
                i += 1
                continue
            j = i
            while j < len(same) and same[j]:
                j += 1
            run = j - i + 1  # includes the first repeated sample at i+1.. 
            # same[k] True means sample k+1 equals sample k; run length in samples ≈ j-i+1
            if 2 <= run <= 15:
                stuck[i + 1 : j + 1] = True
            i = j

    # Spikes relative to local robust scale (skip early protect window later in filter)
    a_norm = np.linalg.norm(accel, axis=1)
    g_norm = np.linalg.norm(gyro, axis=1)
    # Use rolling-ish robust thresholds via percentiles on protected region first
    t0 = t_s[0]
    prot = t_s <= (t0 + protect_seconds)
    if prot.sum() < 10:
        prot = np.ones(n, dtype=bool)
        prot[min(n - 1, 50) :] = False if n > 50 else prot

    a_ref = float(np.median(a_norm[prot])) if prot.any() else 9.81
    g_ref = float(np.median(g_norm[prot])) if prot.any() else 0.0
    a_mad = float(np.median(np.abs(a_norm[prot] - a_ref))) + 1e-6
    g_mad = float(np.median(np.abs(g_norm[prot] - g_ref))) + 1e-6

    # Sensor failure first: trailing region where both norms ~ 0 (degrade_imu zeros last 5s)
    failed = np.zeros(n, dtype=bool)
    near_zero = (a_norm < failure_norm_thresh) & (g_norm < failure_norm_thresh)
    i = n - 1
    while i >= 0 and near_zero[i]:
        i -= 1
    fail_start = i + 1
    if fail_start < n and (t_s[-1] - t_s[fail_start]) >= failure_min_seconds:
        failed[fail_start:] = True
    elif (
        fail_start < n
        and near_zero[fail_start:].mean() > 0.95
        and (n - fail_start) > int(0.5 / med_dt)
    ):
        failed[fail_start:] = True

    spike = (
        (np.abs(a_norm - a_ref) > max(3.5, 10.0 * a_mad))
        | (np.abs(g_norm - g_ref) > max(0.6, 10.0 * g_mad))
    ) & ~failed & ~near_zero

    stuck = stuck & ~failed
    usable = ~(dup | stuck | spike | failed)
    # gaps themselves are not invalid samples — the sample after a gap is usable
    # but we flag gap for coast logic
    return {
        "dt": dt,
        "med_dt": med_dt,
        "dup": dup,
        "gap": gap,
        "stuck": stuck,
        "spike": spike,
        "failed": failed,
        "usable": usable,
        "fail_start_idx": int(fail_start) if failed.any() else n,
        "a_norm": a_norm,
        "g_norm": g_norm,
    }
