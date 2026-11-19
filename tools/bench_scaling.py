#!/usr/bin/env python3
"""Memory/runtime vs trajectory length (incl. 15-min synthetic)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.estimator import run_estimator
from tools.make_synthetic_with_gt import main as _unused  # noqa: F401


def make_long_imu(seconds: float, hz: float = 200.0, out: Path | None = None) -> Path:
    n = int(seconds * hz)
    t0 = 1_400_000_000.0
    t = t0 + np.arange(n) / hz
    # Non-static: gentle yaw rate + gravity in body with small motion accel
    gyro = np.zeros((n, 3))
    gyro[:, 2] = 0.05 * np.sin(2 * np.pi * t / 30.0)
    accel = np.zeros((n, 3))
    accel[:, 2] = 9.80665
    accel[:, 0] = 0.2 * np.sin(2 * np.pi * t / 8.0)
    path = out or (ROOT / "outputs" / "scaling" / f"imu_{int(seconds)}s.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([np.rint(t * 1e9), gyro, accel])
    np.savetxt(path, data, delimiter=",", header="timestamp,wx,wy,wz,ax,ay,az", comments="", fmt="%.6f")
    return path


def main() -> int:
    lengths = [30, 60, 120, 300, 600, 900]  # seconds; 900 = 15 min
    rows = []
    for sec in lengths:
        imu = make_long_imu(sec)
        est = imu.with_name(f"est_{int(sec)}s.txt")
        meta = run_estimator(str(imu), str(est))
        row = {
            "duration_s": sec,
            "n": meta["n_samples"],
            "ms_per_s": meta["ms_per_imu_second"],
            "mb": meta["peak_algorithm_mb"],
            "runtime_s": meta["runtime_s"],
        }
        rows.append(row)
        print(f"{sec:4d}s  n={meta['n_samples']:7d}  {meta['ms_per_imu_second']:.2f} ms/s  {meta['peak_algorithm_mb']:.2f} MB")

    out = ROOT / "outputs" / "scaling" / "scaling.json"
    out.write_text(json.dumps({"rows": rows}, indent=2))
    csv = ROOT / "outputs" / "scaling" / "scaling.csv"
    np.savetxt(
        csv,
        np.array([[r["duration_s"], r["n"], r["ms_per_s"], r["mb"]] for r in rows]),
        delimiter=",",
        header="duration_s,n,ms_per_s,mb",
        comments="",
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        d = [r["duration_s"] for r in rows]
        ax[0].plot(d, [r["ms_per_s"] for r in rows], marker="o")
        ax[0].axhline(5.0, color="gray", ls="--", label="5 ms/s")
        ax[0].set_xlabel("duration (s)")
        ax[0].set_ylabel("ms / IMU-s")
        ax[0].legend()
        ax[1].plot(d, [r["mb"] for r in rows], marker="o")
        ax[1].axhline(2.0, color="gray", ls="--", label="2 MB")
        ax[1].set_xlabel("duration (s)")
        ax[1].set_ylabel("algo MB")
        ax[1].legend()
        fig.suptitle("Scaling: runtime & memory vs length")
        fig.tight_layout()
        fig.savefig(ROOT / "outputs" / "figures" / "scaling.png", dpi=140)
        print("wrote outputs/figures/scaling.png")
    except Exception as ex:
        print("plot skipped", ex)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
