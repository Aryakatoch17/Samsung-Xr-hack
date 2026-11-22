#!/usr/bin/env python3
"""Robustness matrix on a 30 s MH01 slice: seeds × order × max-level × gamma + non-static.

Full-length coverage is the regression gate; this checks format/order/config variants fast.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from degrade_imu import CONFIG, degrade
from src.estimator import run_estimator
from src.main import main as main_cli

SEQ = "MH_01_easy"


def main() -> int:
    seq_dir = ROOT / "data" / "euroc" / SEQ
    imu = seq_dir / "imu.csv"
    work = ROOT / "outputs" / "robustness"
    work.mkdir(parents=True, exist_ok=True)

    raw = np.loadtxt(imu, delimiter=",", comments="#")
    # ~30 s slice for speed
    t0 = raw[0, 0]
    sl = raw[raw[:, 0] <= t0 + 30e9]
    print(f"slice rows={len(sl)}")

    gyro_first = work / "gyro_first.csv"
    accel_first = work / "accel_first.csv"
    np.savetxt(
        gyro_first,
        sl[:, :7],
        delimiter=",",
        header="timestamp,wx,wy,wz,ax,ay,az",
        comments="",
        fmt="%.6f",
    )
    np.savetxt(
        accel_first,
        np.column_stack([sl[:, 0], sl[:, 4:7], sl[:, 1:4]]),
        delimiter=",",
        header="timestamp,ax,ay,az,gx,gy,gz",
        comments="",
        fmt="%.6f",
    )

    rows = []
    for order, src in [("gyro_first", gyro_first), ("accel_first", accel_first)]:
        df = pd.read_csv(src)
        for seed in (42, 7, 99):
            for max_level in (0.5, 1.0):
                for gamma in (1.0, 1.7):
                    cfg = deepcopy(CONFIG)
                    cfg["max_level"] = max_level
                    cfg["gamma"] = gamma
                    out_csv = work / f"{order}_s{seed}_m{max_level}_g{gamma}.csv"
                    deg, _ = degrade(df, cfg, seed)
                    deg.to_csv(out_csv, index=False)
                    est = out_csv.with_suffix(".txt")
                    meta = run_estimator(str(out_csv), str(est))
                    ok = meta["n_samples"] == len(deg) and meta["ms_per_imu_second"] <= 10.0
                    rows.append(
                        {
                            "order": order,
                            "seed": seed,
                            "max_level": max_level,
                            "gamma": gamma,
                            "n": meta["n_samples"],
                            "ms_per_s": round(meta["ms_per_imu_second"], 3),
                            "mb": round(meta["peak_algorithm_mb"], 3),
                            "ok": ok,
                        }
                    )
                    print(
                        f"{order} seed={seed} L={max_level} g={gamma} "
                        f"n={meta['n_samples']} {meta['ms_per_imu_second']:.1f} ms/s ok={ok}"
                    )

    # Non-static start: drop first 5 s of slice
    cut = sl[sl[:, 0] >= t0 + 5e9]
    ns_path = work / "nonstatic_accel_first.csv"
    np.savetxt(
        ns_path,
        np.column_stack([cut[:, 0], cut[:, 4:7], cut[:, 1:4]]),
        delimiter=",",
        header="timestamp,ax,ay,az,gx,gy,gz",
        comments="",
        fmt="%.6f",
    )
    df = pd.read_csv(ns_path)
    deg, _ = degrade(df, deepcopy(CONFIG), 42)
    deg_path = work / "nonstatic_degraded.csv"
    deg.to_csv(deg_path, index=False)
    est = work / "nonstatic_est.txt"
    meta = run_estimator(str(deg_path), str(est))
    rows.append(
        {
            "order": "nonstatic_accel_first",
            "seed": 42,
            "n": meta["n_samples"],
            "ms_per_s": round(meta["ms_per_imu_second"], 3),
            "ok": meta["n_samples"] == len(deg),
        }
    )

    rc = main_cli(
        [str(accel_first), "--out", str(work / "cli_out.txt"), "--meta", str(work / "cli_meta.json")]
    )
    rows.append({"order": "cli_smoke", "ok": rc == 0})

    # Rate / format stress (hidden-file generalization).
    base = sl[:, :7].copy()
    t_ns = base[:, 0].astype(np.float64)
    t0n = float(t_ns[0])
    span_s = (float(t_ns[-1]) - t0n) * 1e-9

    def _run_stress(name: str, arr: np.ndarray, *, header: str | None, nl: str = "\n") -> None:
        p = work / f"stress_{name}.csv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            if header:
                f.write(header + nl)
            for row in arr:
                f.write(
                    (",".join(f"{x:.9f}" if np.isfinite(x) else "nan" for x in row)) + nl
                )
        est = p.with_suffix(".txt")
        try:
            meta = run_estimator(str(p), str(est), measure_tracemalloc=False)
            # Short files (<5 s) are dominated by fixed I/O; FAQ ms/s is for long IMU.
            ok = meta["n_samples"] >= 10 and (
                float(meta["duration_s"]) < 5.0 or meta["ms_per_imu_second"] <= 20.0
            )
        except Exception as exc:  # noqa: BLE001
            meta = {"error": str(exc)}
            ok = False
        rows.append({"order": f"stress_{name}", "ok": ok, "meta": meta})
        print(f"stress_{name} ok={ok}")

    # 100 Hz resample of slice
    n100 = max(20, int(span_s * 100))
    t100 = np.linspace(t0n, t_ns[-1], n100)
    a100 = np.column_stack([t100] + [np.interp(t100, t_ns, base[:, j]) for j in range(1, 7)])
    _run_stress("100hz", a100, header="timestamp,wx,wy,wz,ax,ay,az")

    # 1 kHz (short 2 s to keep runtime bounded)
    sl2 = base[t_ns <= t0n + 2e9]
    if len(sl2) >= 10:
        t2 = sl2[:, 0]
        span2 = (float(t2[-1]) - float(t2[0])) * 1e-9
        n1k = max(20, int(span2 * 1000))
        t1k = np.linspace(t2[0], t2[-1], n1k)
        a1k = np.column_stack([t1k] + [np.interp(t1k, t2, sl2[:, j]) for j in range(1, 7)])
        _run_stress("1khz", a1k, header="timestamp,wx,wy,wz,ax,ay,az")

    # Microsecond timestamps
    us = base.copy()
    us[:, 0] = (t_ns - t0n) * 1e-3  # ns → µs from start
    _run_stress("timestamps_us", us, header="timestamp,wx,wy,wz,ax,ay,az")

    # No header
    _run_stress("no_header", base, header=None)

    # CRLF line endings
    _run_stress("crlf", base, header="timestamp,wx,wy,wz,ax,ay,az", nl="\r\n")

    # NaN rows (inject 5)
    nan_arr = base.copy()
    for i in (10, 50, 100, 200, 400):
        if i < len(nan_arr):
            nan_arr[i, 1:] = np.nan
    _run_stress("nan_rows", nan_arr, header="timestamp,wx,wy,wz,ax,ay,az")

    # Dead gyro-Z channel
    dead = base.copy()
    dead[:, 3] = 0.0
    _run_stress("dead_gz", dead, header="timestamp,wx,wy,wz,ax,ay,az")

    out = {"rows": rows, "n_ok": sum(1 for r in rows if r.get("ok")), "n": len(rows)}
    (work / "robustness_matrix.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {work / 'robustness_matrix.json'} ok={out['n_ok']}/{out['n']}")
    return 0 if out["n_ok"] == out["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
