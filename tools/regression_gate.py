#!/usr/bin/env python3
"""
Regression gate vs v1-safe targets.

Fails if (any seed, any sequence):
  ATE or RPE-trans > 1.05x stay-put
  last-5s > 1.10x stay-put
  RPE-rot or ATE-rot gain drops >10% vs locked baseline
  runtime > 10 ms/s
  algo account heap > 5 MB (FAQ ≤8; tracemalloc/RSS logged separately)

Usage:
  python tools/regression_gate.py [--seeds 42,7,99] [--seqs MH_01_easy,...]
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from degrade_imu import CONFIG, SEED, degrade  # noqa: E402
from metrics_lib import evaluate_pair  # noqa: E402
from src.estimator import run_estimator  # noqa: E402

# Locked at v1-safe on *degraded* IMU (seed 42). Gate fails if gain drops >10%.
# (Clean-IMU gains look higher; do not use those as the regression floor.)
BASELINE_ROT = {
    "MH_01_easy": {"rpe_rot_gain": 16.0, "ate_rot_gain": 4.5},
    "MH_02_easy": {"rpe_rot_gain": 7.0, "ate_rot_gain": 5.5},
    "MH_03_medium": {"rpe_rot_gain": 18.0, "ate_rot_gain": 8.0},
    "MH_04_difficult": {"rpe_rot_gain": 14.0, "ate_rot_gain": 6.0},
    "MH_05_difficult": {"rpe_rot_gain": 20.0, "ate_rot_gain": 10.0},
}

DEFAULT_SEQS = list(BASELINE_ROT.keys())


def _accel_first_csv(imu_path: Path, out_path: Path) -> None:
    raw = np.loadtxt(imu_path, delimiter=",", comments="#")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    # EuRoC: ts,gx,gy,gz,ax,ay,az -> accel-first
    out = np.column_stack([raw[:, 0], raw[:, 4:7], raw[:, 1:4]])
    np.savetxt(out_path, out, delimiter=",", header="timestamp,ax,ay,az,gx,gy,gz", comments="", fmt="%.6f")


def degrade_to_file(accel_first: Path, out_csv: Path, seed: int, cfg: dict) -> None:
    df = pd.read_csv(accel_first)
    out, _stats = degrade(df, cfg, seed)
    out.to_csv(out_csv, index=False)


def check_row(m: dict, seq: str, ms_per_s: float, peak_mb: float) -> list[str]:
    fails = []
    if m["ate"] > 1.05 * m["stayput_ate"]:
        fails.append(f"ATE {m['ate']:.3f} > 1.05x SP {m['stayput_ate']:.3f}")
    if m["rpe_1s"] > 1.05 * m["stayput_rpe"]:
        fails.append(f"RPE {m['rpe_1s']:.3f} > 1.05x SP {m['stayput_rpe']:.3f}")
    if m["blind5"] > 1.10 * m["stayput_blind5"]:
        fails.append(f"B5 {m['blind5']:.3f} > 1.10x SP {m['stayput_blind5']:.3f}")
    base = BASELINE_ROT[seq]
    if m["rpe_rot_gain"] < 0.9 * base["rpe_rot_gain"]:
        fails.append(
            f"RPE-rot gain {m['rpe_rot_gain']:.1f} < 90% of baseline {base['rpe_rot_gain']:.1f}"
        )
    if m["ate_rot_gain"] < 0.9 * base["ate_rot_gain"]:
        fails.append(
            f"ATE-rot gain {m['ate_rot_gain']:.1f} < 90% of baseline {base['ate_rot_gain']:.1f}"
        )
    if ms_per_s > 10.0:
        fails.append(f"runtime {ms_per_s:.2f} ms/s > 10")
    if peak_mb > 5.0:
        fails.append(f"algo account {peak_mb:.2f} MB > 5")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,99")
    ap.add_argument("--seqs", default=",".join(DEFAULT_SEQS))
    ap.add_argument("--out", default="outputs/regression_gate.json")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    seqs = [s.strip() for s in args.seqs.split(",") if s.strip()]

    rows = []
    all_fails: list[str] = []
    cfg = deepcopy(CONFIG)

    for seq in seqs:
        seq_dir = ROOT / "data" / "euroc" / seq
        imu = seq_dir / "imu.csv"
        gt = ROOT / "outputs" / seq / "gt_tum.txt"
        if not imu.exists() or not gt.exists():
            all_fails.append(f"{seq}: missing imu or gt")
            continue
        work = ROOT / "outputs" / "gate" / seq
        work.mkdir(parents=True, exist_ok=True)
        accel_first = work / "imu_accel_first.csv"
        _accel_first_csv(imu, accel_first)

        for seed in seeds:
            deg = work / f"imu_seed{seed}.csv"
            est = work / f"est_seed{seed}.txt"
            degrade_to_file(accel_first, deg, seed, cfg)

            # Timed pass: no internal tracemalloc second-pass (keeps ms/s honest).
            meta = run_estimator(str(deg), str(est), measure_tracemalloc=False)
            ms = float(meta["ms_per_imu_second"])
            account_mb = float(meta["peak_algorithm_mb"])
            rss_mb = meta.get("rss_delta_mb")
            # One dual-heap sample per sequence (seed 42 only).
            trace_mb = float("nan")
            if seed == seeds[0]:
                meta_tm = run_estimator(
                    str(deg),
                    str(work / f"_trace_seed{seed}.txt"),
                    measure_tracemalloc=True,
                )
                trace_mb = float(meta_tm.get("tracemalloc_peak_mb") or float("nan"))
                if rss_mb is None:
                    rss_mb = meta_tm.get("rss_delta_mb")

            m = evaluate_pair(str(gt), str(est))
            fails = check_row(m, seq, ms, account_mb)  # FAQ algorithm heap
            row = {
                "seq": seq,
                "seed": seed,
                "ms_per_s": round(ms, 3),
                "tracemalloc_peak_mb": round(trace_mb, 3) if trace_mb == trace_mb else None,
                "rss_delta_mb": None if rss_mb is None else round(float(rss_mb), 3),
                "algo_account_mb": round(account_mb, 3),
                "sensor_storage": meta.get("sensor_storage"),
                **{k: round(float(v), 4) for k, v in m.items()},
                "fails": fails,
                "pass": len(fails) == 0,
            }
            rows.append(row)
            status = "PASS" if row["pass"] else "FAIL"
            print(
                f"[{status}] {seq} seed={seed} ATE={m['ate']:.3f}/{m['stayput_ate']:.3f} "
                f"RPE={m['rpe_1s']:.3f} B5={m['blind5']:.3f} "
                f"Rrot={m['rpe_rot_deg']:.2f}° (gain {m['rpe_rot_gain']:.1f}x) "
                f"{ms:.2f} ms/s  acct={account_mb:.2f} MB  "
                f"tm={trace_mb if trace_mb == trace_mb else float('nan'):.2f} MB  "
                f"rssΔ={rss_mb if rss_mb is not None else float('nan'):.2f} MB"
            )
            for f in fails:
                print(f"    - {f}")
                all_fails.append(f"{seq}/seed{seed}: {f}")

    out = {
        "seeds": seeds,
        "n_rows": len(rows),
        "n_pass": sum(1 for r in rows if r["pass"]),
        "n_fail": sum(1 for r in rows if not r["pass"]),
        "rows": rows,
        "failures": all_fails,
        "official_seed": SEED,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}  pass={out['n_pass']}/{out['n_rows']}")
    return 0 if out["n_fail"] == 0 and len(rows) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
