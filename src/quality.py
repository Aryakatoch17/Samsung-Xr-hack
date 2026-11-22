"""Sample quality flags: spikes, stuck, duplicates, gaps, sensor failure."""

from __future__ import annotations

import numpy as np

_CHUNK = 8192


def analyze_imu(
    t_s: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    protect_seconds: float = 2.0,
    failure_norm_thresh: float = 0.15,
    failure_min_seconds: float = 1.0,
) -> dict:
    n = len(t_s)
    # dt without a lasting full copy beyond this function's peak overlap.
    if n > 1:
        d = np.diff(t_s)
        med_dt = float(np.nanmedian(d))
        if not np.isfinite(med_dt) or med_dt <= 0:
            med_dt = 0.005
        med_dt = max(med_dt, 1e-4)
        dup = np.zeros(n, dtype=bool)
        gap = np.zeros(n, dtype=bool)
        dup[1:] = d <= 1e-9
        gap[1:] = d > (3.5 * med_dt)
        del d
    else:
        med_dt = 0.005
        dup = np.zeros(n, dtype=bool)
        gap = np.zeros(n, dtype=bool)

    # Stuck: chunked pairwise compare — never allocate full (n,3) diff.
    stuck = np.zeros(n, dtype=bool)
    if n > 1:
        for start in range(0, n - 1, _CHUNK):
            end = min(n - 1, start + _CHUNK)
            same = np.all(
                np.abs(accel[start + 1 : end + 1] - accel[start:end]) < 1e-9, axis=1
            ) & np.all(
                np.abs(gyro[start + 1 : end + 1] - gyro[start:end]) < 1e-9, axis=1
            )
            i = 0
            while i < len(same):
                if not same[i]:
                    i += 1
                    continue
                j = i
                while j < len(same) and same[j]:
                    j += 1
                run = j - i + 1
                if 2 <= run <= 15:
                    stuck[start + i + 1 : start + j + 1] = True
                i = j

    # One float32 norm buffer at a time (reuse) to avoid 2×N live norms.
    norm = np.empty(n, dtype=np.float32)
    for start in range(0, n, _CHUNK):
        end = min(n, start + _CHUNK)
        norm[start:end] = np.linalg.norm(accel[start:end], axis=1)

    t0 = t_s[0]
    n_prot = int(np.searchsorted(t_s, t0 + protect_seconds, side="right"))
    if n_prot < 10:
        n_prot = min(n, 50)
    prot_sl = slice(0, n_prot)

    a_ref = float(np.median(norm[prot_sl])) if n_prot else 9.81
    a_mad = float(np.median(np.abs(norm[prot_sl] - a_ref))) + 1e-6
    a_spike = np.abs(norm - a_ref) > max(3.5, 10.0 * a_mad)
    near_zero_a = norm < failure_norm_thresh

    for start in range(0, n, _CHUNK):
        end = min(n, start + _CHUNK)
        norm[start:end] = np.linalg.norm(gyro[start:end], axis=1)
    g_ref = float(np.median(norm[prot_sl])) if n_prot else 0.0
    g_mad = float(np.median(np.abs(norm[prot_sl] - g_ref))) + 1e-6
    g_spike = np.abs(norm - g_ref) > max(0.6, 10.0 * g_mad)
    near_zero = near_zero_a & (norm < failure_norm_thresh)
    del near_zero_a

    failed = np.zeros(n, dtype=bool)
    i = n - 1
    while i >= 0 and near_zero[i]:
        i -= 1
    fail_start = i + 1
    if fail_start < n and (t_s[-1] - t_s[fail_start]) >= failure_min_seconds:
        failed[fail_start:] = True
    elif (
        fail_start < n
        and float(near_zero[fail_start:].mean()) > 0.95
        and (n - fail_start) > int(0.5 / med_dt)
    ):
        failed[fail_start:] = True

    spike = (a_spike | g_spike) & ~failed & ~near_zero
    del a_spike, g_spike, near_zero, norm

    stuck = stuck & ~failed
    usable = ~(dup | stuck | spike | failed)
    return {
        "med_dt": med_dt,
        "dup": dup,
        "gap": gap,
        "stuck": stuck,
        "spike": spike,
        "failed": failed,
        "usable": usable,
        "fail_start_idx": int(fail_start) if failed.any() else n,
    }
