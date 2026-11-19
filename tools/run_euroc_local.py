#!/usr/bin/env python3
"""
Run the Track-3 pipeline on a local EuRoC extract.

Expected:
  data/euroc/<seq>/imu.csv   (EuRoC imu0/data.csv)
  data/euroc/<seq>/gt.csv    (state_groundtruth_estimate0/data.csv)  [optional for scoring]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True, help="Folder with imu.csv (+ optional gt.csv)")
    ap.add_argument("--max-level", type=float, default=1.0)
    args = ap.parse_args()

    seq = Path(args.seq_dir)
    if not seq.is_absolute():
        seq = ROOT / seq
    imu = seq / "imu.csv"
    if not imu.exists():
        raise SystemExit(f"missing {imu}")

    out = ROOT / "outputs" / seq.name
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    accel_first = out / "imu_accel_first.csv"
    degraded = out / "imu_degraded.csv"
    est = out / "estimated_trajectory.txt"
    meta = out / "metadata.json"

    run([py, "tools/euroc_to_degrade_input.py", str(imu), str(accel_first)])
    run([py, "degrade_imu.py", str(accel_first), str(degraded), "--max-level", str(args.max_level)])
    run([py, "src/main.py", str(degraded), "--out", str(est), "--meta", str(meta)])

    gt_csv = seq / "gt.csv"
    if gt_csv.exists():
        gt_tum = out / "gt_tum.txt"
        run([py, "tools/euroc_gt_to_tum.py", str(gt_csv), str(gt_tum)])
        run([py, "tools/score_evo.py", str(gt_tum), str(est), "--out-dir", str(out / "evo"), "--no-plot"])
    else:
        print("no gt.csv — skipped evo scoring")
    print(f"done: {est}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
