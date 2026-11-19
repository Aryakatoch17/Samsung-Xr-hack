#!/usr/bin/env python3
"""Generate synthetic data, degrade, estimate, and print evo ATE."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def score(gt: Path, est: Path, label: str) -> float:
    from evo.core import metrics, sync
    from evo.tools import file_interface

    g = file_interface.read_tum_trajectory_file(str(gt))
    e = file_interface.read_tum_trajectory_file(str(est))
    g, e = sync.associate_trajectories(g, e, max_diff=0.02)
    e.align(g, correct_scale=False)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((g, e))
    rmse = float(ape.get_all_statistics()["rmse"])
    print(f"{label} ATE_RMSE={rmse:.4f} m")
    return rmse


def main() -> int:
    py = sys.executable
    run([py, "tools/make_synthetic_with_gt.py", "--out-dir", "data/synthetic", "--seconds", "40"])
    run([py, "tools/euroc_to_degrade_input.py", "data/synthetic/imu_euroc.csv", "data/synthetic/imu_accel_first.csv"])
    run([py, "degrade_imu.py", "data/synthetic/imu_accel_first.csv", "data/synthetic/imu_degraded.csv"])
    run([py, "src/main.py", "data/synthetic/imu_euroc.csv", "--out", "outputs/synthetic/est_clean.txt", "--meta", "outputs/synthetic/meta_clean.json"])
    run([py, "src/main.py", "data/synthetic/imu_degraded.csv", "--out", "outputs/synthetic/estimated_trajectory.txt", "--meta", "outputs/synthetic/metadata.json"])
    clean = score(ROOT / "data/synthetic/gt_tum.txt", ROOT / "outputs/synthetic/est_clean.txt", "CLEAN")
    degr = score(ROOT / "data/synthetic/gt_tum.txt", ROOT / "outputs/synthetic/estimated_trajectory.txt", "DEGRADED")
    meta = json.loads((ROOT / "outputs/synthetic/metadata.json").read_text(encoding="utf-8"))
    print(
        f"runtime={meta['ms_per_imu_second']:.2f} ms/s  "
        f"algo_mem={meta['peak_algorithm_mb']:.2f} MB  "
        f"failed={meta['n_failed']} spikes={meta['n_spikes']} gaps={meta['n_gaps']}"
    )
    out = {"clean_ate_rmse": clean, "degraded_ate_rmse": degr, "meta": meta}
    (ROOT / "outputs/synthetic/benchmark.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
