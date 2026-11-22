#!/usr/bin/env python3
"""One-shot held-out V1/V2 verify (degraded seed 42)."""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from degrade_imu import CONFIG, degrade
from metrics_lib import evaluate_pair
from src.estimator import run_estimator


def main() -> int:
    all_ok = True
    for name in ("V1_01_easy", "V2_01_easy"):
        seq = ROOT / "data" / "euroc" / name
        imu = seq / "imu.csv"
        gt = seq / "gt_tum.txt"
        df = pd.read_csv(imu)
        deg, _ = degrade(df, deepcopy(CONFIG), 42)
        od = ROOT / "outputs" / "holdout_verify" / name
        od.mkdir(parents=True, exist_ok=True)
        deg_path = od / "imu_degraded.csv"
        deg.to_csv(deg_path, index=False)
        est = od / "est.txt"
        meta = run_estimator(str(deg_path), str(est), measure_tracemalloc=True)
        m = evaluate_pair(str(gt), str(est))
        ratio = m["ate"] / max(m["stayput_ate"], 1e-9)
        rpe_x = m["rpe_1s"] / max(m["stayput_rpe"], 1e-9)
        b5_x = m["blind5"] / max(m["stayput_blind5"], 1e-9)
        ok = (
            ratio <= 1.05
            and rpe_x <= 1.05
            and b5_x <= 1.10
            and meta["ms_per_imu_second"] <= 10
            and (meta.get("tracemalloc_peak_mb") or 0) <= 8
        )
        all_ok = all_ok and ok
        print(
            f"{name} ATE={m['ate']:.3f}/{m['stayput_ate']:.3f} ({ratio:.3f}) "
            f"RPE×={rpe_x:.3f} B5×={b5_x:.3f} Rrot={m['rpe_rot_deg']:.2f}° "
            f"ms={meta['ms_per_imu_second']:.2f} "
            f"traced={meta.get('ms_per_imu_second_traced'):.2f} "
            f"tm={meta.get('tracemalloc_peak_mb'):.2f} OK={ok}"
        )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
