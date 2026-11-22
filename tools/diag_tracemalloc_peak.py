#!/usr/bin/env python3
"""Warm imports, then tracemalloc the estimator — top-10 at peak."""

from __future__ import annotations

import json
import sys
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- imports BEFORE tracing (protocol: heap pass starts after numpy is loaded) ---
import numpy as np  # noqa: E402

from src.estimator import ATT_HZ, VEL_HP_TAU_S, _run_core  # noqa: E402
from src.io_utils import load_imu_csv, write_tum_from_keyframes  # noqa: E402
from src.fast_ins import run_lean_ins  # noqa: E402
from src.quality import analyze_imu  # noqa: E402
from src.mem_utils import cleanup_spill  # noqa: E402


def _top10(snap: tracemalloc.Snapshot) -> list[dict]:
    rows = []
    for s in snap.statistics("lineno")[:10]:
        frame = s.traceback[0] if s.traceback else None
        rows.append(
            {
                "mb": round(s.size / (1024 * 1024), 4),
                "count": s.count,
                "file": getattr(frame, "filename", "?"),
                "line": getattr(frame, "lineno", -1),
            }
        )
    return rows


def main() -> int:
    imu = ROOT / "data" / "euroc" / "MH_01_easy" / "imu.csv"
    out = ROOT / "outputs" / "diag" / "tm_peak_traj.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Warm path once so module-level caches exist.
    _run_core(
        str(imu),
        str(out.with_name("warm.txt")),
        vel_hp_tau_s=VEL_HP_TAU_S,
        att_hz=ATT_HZ,
        gravity_aid=False,
        gravity_kp=0.05,
    )

    phases: list[dict] = []
    peak_hold = {"bytes": 0, "phase": "", "top10": []}

    def note(phase: str) -> None:
        cur, peak = tracemalloc.get_traced_memory()
        snap = tracemalloc.take_snapshot()
        top = _top10(snap)
        phases.append(
            {
                "phase": phase,
                "cur_mb": cur / (1024 * 1024),
                "peak_mb": peak / (1024 * 1024),
                "top10_now": top,
            }
        )
        if peak >= peak_hold["bytes"]:
            peak_hold["bytes"] = peak
            peak_hold["phase"] = phase
            peak_hold["top10"] = top

    tracemalloc.start(25)
    note("start_after_imports")

    imu_d = load_imu_csv(str(imu))
    note("after_load_imu_csv")
    t_s = imu_d["t_s"]
    gyro = imu_d["gyro"]
    accel = imu_d["accel"]
    tmp = imu_d.get("sensor_tmp")
    n = imu_d["n"]
    del imu_d

    qual = analyze_imu(t_s, accel, gyro)
    note("after_analyze_imu")
    usable, failed, gap, dup = qual["usable"], qual["failed"], qual["gap"], qual["dup"]
    del qual

    lean = run_lean_ins(
        t_s, gyro, accel, usable, failed, gap, dup, vel_hp_tau_s=VEL_HP_TAU_S, att_hz=ATT_HZ
    )
    note("after_run_lean_ins")

    write_tum_from_keyframes(
        str(out), t_s, lean["t_d"], lean["pos_d"], lean["quat_d"], failed, dup
    )
    note("after_write_tum")
    cleanup_spill(tmp)
    note("after_cleanup")

    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    report = {
        "protocol": "tracemalloc started AFTER import numpy / src.* (warm run first)",
        "n_samples": n,
        "final_peak_mb": peak / (1024 * 1024),
        "peak_observed_at_phase": peak_hold["phase"],
        "peak_top10": peak_hold["top10"],
        "phases": phases,
    }
    out_json = ROOT / "outputs" / "diag" / "tracemalloc_peak_MH01.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("final_peak_mb", "peak_observed_at_phase", "peak_top10")}, indent=2))
    print("wrote", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
