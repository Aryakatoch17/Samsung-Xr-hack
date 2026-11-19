#!/usr/bin/env python3
"""
Track 03 – Path Finder
Standalone entrypoint: degraded IMU CSV -> TUM trajectory.

Usage:
    python -m src.main <imu.csv> [--out estimated_trajectory.txt] [--meta metadata.json]
    python src/main.py <imu.csv> [--out estimated_trajectory.txt] [--meta metadata.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python src/main.py ...` without installing the package
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.estimator import run_estimator  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Track 3 Path Finder – IMU trajectory estimator")
    ap.add_argument("imu_csv", help="Path to degraded IMU CSV (EuRoC or accel-first)")
    ap.add_argument(
        "--out",
        default="estimated_trajectory.txt",
        help="Output TUM trajectory path (default: estimated_trajectory.txt)",
    )
    ap.add_argument(
        "--meta",
        default="metadata.json",
        help="Optional metadata JSON path (default: metadata.json)",
    )
    args = ap.parse_args(argv)

    imu_path = Path(args.imu_csv)
    if not imu_path.exists():
        print(f"error: input not found: {imu_path}", file=sys.stderr)
        return 1

    meta = run_estimator(str(imu_path), args.out)
    if args.meta:
        with open(args.meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print(f"wrote {args.out}")
    print(
        f"  samples={meta['n_samples']}  duration={meta['duration_s']:.2f}s  "
        f"runtime={meta['runtime_s']:.3f}s  ms/s={meta['ms_per_imu_second']:.2f}"
    )
    print(
        f"  algo_mem≈{meta['peak_algorithm_mb']:.2f} MB  "
        f"failed={meta['n_failed']}  spikes={meta['n_spikes']}  gaps={meta['n_gaps']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
