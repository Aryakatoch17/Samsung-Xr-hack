#!/usr/bin/env python3
"""Memory/runtime vs trajectory length — tracemalloc + RSS + account."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.estimator import run_estimator


def make_long_imu(seconds: float, hz: float = 200.0, out: Path | None = None) -> Path:
    n = int(seconds * hz)
    t0 = 1_400_000_000.0
    t = t0 + np.arange(n) / hz
    gyro = np.zeros((n, 3))
    gyro[:, 2] = 0.05 * np.sin(2 * np.pi * t / 30.0)
    accel = np.zeros((n, 3))
    accel[:, 2] = 9.80665
    accel[:, 0] = 0.2 * np.sin(2 * np.pi * t / 8.0)
    path = out or (ROOT / "outputs" / "scaling" / f"imu_{int(seconds)}s.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([np.rint(t * 1e9), gyro, accel])
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="timestamp,wx,wy,wz,ax,ay,az",
        comments="",
        fmt="%.6f",
    )
    return path


def main() -> int:
    lengths = [30, 60, 120, 300, 600, 900]  # seconds; 900 = 15 min
    rows = []
    for sec in lengths:
        imu = make_long_imu(sec)
        est = imu.with_name(f"est_{int(sec)}s.txt")
        meta = run_estimator(str(imu), str(est), measure_tracemalloc=True)
        row = {
            "duration_s": sec,
            "n": meta["n_samples"],
            "ms_per_s": meta["ms_per_imu_second"],
            "ms_per_s_traced": meta.get("ms_per_imu_second_traced"),
            "account_mb": meta["peak_algorithm_mb"],
            "tracemalloc_mb": meta.get("tracemalloc_peak_mb"),
            "rss_delta_mb": meta.get("rss_delta_mb"),
            "sensor_storage": meta.get("sensor_storage"),
            "runtime_s": meta["runtime_s"],
        }
        rows.append(row)
        print(
            f"{sec:4d}s  n={meta['n_samples']:7d}  "
            f"ms/s={meta['ms_per_imu_second']:.2f}  "
            f"ms/s_tm={meta.get('ms_per_imu_second_traced')}  "
            f"tm={meta.get('tracemalloc_peak_mb')}  "
            f"acct={meta['peak_algorithm_mb']:.2f}  "
            f"rssΔ={meta.get('rss_delta_mb')}  "
            f"store={meta.get('sensor_storage')}"
        )

    out = ROOT / "outputs" / "scaling" / "scaling.json"
    out.write_text(json.dumps({"rows": rows}, indent=2))
    csv = ROOT / "outputs" / "scaling" / "scaling.csv"
    with open(csv, "w", encoding="utf-8") as f:
        f.write(
            "duration_s,n,ms_per_s,ms_per_s_traced,tracemalloc_mb,account_mb,rss_delta_mb\n"
        )
        for r in rows:
            f.write(
                f"{r['duration_s']},{r['n']},{r['ms_per_s']},"
                f"{r['ms_per_s_traced']},{r['tracemalloc_mb']},"
                f"{r['account_mb']},{r['rss_delta_mb']}\n"
            )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        d = [r["duration_s"] for r in rows]
        ax[0].plot(d, [r["ms_per_s"] for r in rows], marker="o", label="ms/s")
        ax[0].plot(
            d,
            [r["ms_per_s_traced"] or np.nan for r in rows],
            marker="s",
            label="ms/s + tracemalloc",
        )
        ax[0].axhline(20.0, color="red", ls="--", label="20 ms/s cap")
        ax[0].axhline(5.0, color="gray", ls=":", label="5 ms/s")
        ax[0].set_xlabel("duration (s)")
        ax[0].set_ylabel("ms / IMU-s")
        ax[0].legend(fontsize=8)
        ax[1].plot(
            d,
            [r["tracemalloc_mb"] or np.nan for r in rows],
            marker="o",
            label="tracemalloc (headline)",
        )
        ax[1].plot(d, [r["account_mb"] for r in rows], marker="^", label="account")
        ax[1].plot(
            d,
            [r["rss_delta_mb"] if r["rss_delta_mb"] is not None else np.nan for r in rows],
            marker="s",
            label="RSS Δ",
        )
        ax[1].axhline(8.0, color="red", ls="--", label="8 MB cap")
        ax[1].set_xlabel("duration (s)")
        ax[1].set_ylabel("MB")
        ax[1].legend(fontsize=8)
        fig.suptitle("Scaling: runtime & memory (tracemalloc / account / RSS)")
        fig.tight_layout()
        fig.savefig(ROOT / "outputs" / "figures" / "scaling.png", dpi=140)
        print("wrote outputs/figures/scaling.png")
    except Exception as ex:
        print("plot skipped", ex)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
