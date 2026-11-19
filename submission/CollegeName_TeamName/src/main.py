#!/usr/bin/env python3
"""
Track 03 – Path Finder
Standalone entrypoint: degraded IMU CSV -> TUM trajectory.

Accepts:
  - a degraded IMU CSV (EuRoC or accel-first order)
  - a folder containing imu0/data.csv (or mav0/imu0/data.csv)
  - a zip with that layout

Any exception still writes a valid hold-pose trajectory (one row per input row).

Usage:
    python -m src.main <imu.csv|folder|zip> [--out estimated_trajectory.txt] [--meta metadata.json]
    python src/main.py <imu.csv|folder|zip> ...
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.estimator import run_estimator, run_hold_pose_fallback  # noqa: E402
from src.io_utils import resolve_imu_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Track 3 Path Finder – IMU trajectory estimator")
    ap.add_argument("imu_csv", help="Degraded IMU CSV, folder with imu0/data.csv, or zip")
    ap.add_argument("--out", default="estimated_trajectory.txt")
    ap.add_argument("--meta", default="metadata.json")
    ap.add_argument("--tau", type=float, default=None, help="Override velocity HP tau (s)")
    ap.add_argument("--att-hz", type=float, default=None, help="Attitude loop rate (Hz)")
    ap.add_argument("--gravity-aid", action="store_true", help="Low-gain gated gravity on attitude")
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        imu_path = resolve_imu_path(args.imu_csv)
    except Exception as e:
        print(f"error: cannot resolve IMU path: {e}", file=sys.stderr)
        return 1

    kwargs = {}
    if args.tau is not None:
        kwargs["vel_hp_tau_s"] = float(args.tau)
    if args.att_hz is not None:
        kwargs["att_hz"] = float(args.att_hz)
    if args.gravity_aid:
        kwargs["gravity_aid"] = True

    try:
        meta = run_estimator(str(imu_path), str(out_path), **kwargs)
    except Exception as e:
        traceback.print_exc()
        print(f"estimator failed ({e}); writing hold-pose fallback", file=sys.stderr)
        try:
            meta = run_hold_pose_fallback(str(imu_path), str(out_path), err=str(e))
        except Exception as e2:
            print(f"fallback also failed: {e2}", file=sys.stderr)
            return 2

    if args.meta:
        with open(args.meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print(f"wrote {out_path}")
    print(
        f"  samples={meta.get('n_samples')}  duration={meta.get('duration_s', 0):.2f}s  "
        f"runtime={meta.get('runtime_s', 0):.3f}s  ms/s={meta.get('ms_per_imu_second', 0):.2f}"
    )
    print(
        f"  algo_mem≈{meta.get('peak_algorithm_mb', 0):.2f} MB  "
        f"failed={meta.get('n_failed', 0)}  spikes={meta.get('n_spikes', 0)}  "
        f"gaps={meta.get('n_gaps', 0)}  fallback={meta.get('fallback', False)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
