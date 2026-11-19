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

    out = {"rows": rows, "n_ok": sum(1 for r in rows if r.get("ok")), "n": len(rows)}
    (work / "robustness_matrix.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {work / 'robustness_matrix.json'} ok={out['n_ok']}/{out['n']}")
    return 0 if out["n_ok"] == out["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
