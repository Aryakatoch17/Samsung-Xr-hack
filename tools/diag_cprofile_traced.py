#!/usr/bin/env python3
"""cProfile a tracemalloc-on estimator pass (runtime exposure)."""

from __future__ import annotations

import cProfile
import pstats
import sys
import time
import tracemalloc
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402 — imports BEFORE tracing

from src.estimator import ATT_HZ, VEL_HP_TAU_S, _run_core  # noqa: E402


def main() -> int:
    imu = str(ROOT / "data" / "euroc" / "MH_01_easy" / "imu.csv")
    out = str(ROOT / "outputs" / "diag" / "cprofile_traj.txt")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    # Warm untraced
    _run_core(imu, out, vel_hp_tau_s=VEL_HP_TAU_S, att_hz=ATT_HZ, gravity_aid=False, gravity_kp=0.05)

    # Timed untraced
    t0 = time.perf_counter()
    fields, _, _ = _run_core(
        imu, out, vel_hp_tau_s=VEL_HP_TAU_S, att_hz=ATT_HZ, gravity_aid=False, gravity_kp=0.05
    )
    ms = (time.perf_counter() - t0) * 1000.0 / max(float(fields["duration_s"]), 1e-6)

    # Traced + cProfile
    pr = cProfile.Profile()
    tracemalloc.start()
    pr.enable()
    t1 = time.perf_counter()
    _run_core(
        imu,
        out + ".tm.txt",
        vel_hp_tau_s=VEL_HP_TAU_S,
        att_hz=ATT_HZ,
        gravity_aid=False,
        gravity_kp=0.05,
    )
    elapsed_tm = time.perf_counter() - t1
    pr.disable()
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ms_tm = elapsed_tm * 1000.0 / max(float(fields["duration_s"]), 1e-6)

    buf = StringIO()
    stats = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    stats.print_stats(30)
    text = buf.getvalue()
    out_p = ROOT / "outputs" / "diag" / "cprofile_traced_MH01.txt"
    out_p.write_text(
        f"ms_per_s_untraced={ms:.3f}\nms_per_s_traced={ms_tm:.3f}\n"
        f"tracemalloc_peak_mb={peak/(1024*1024):.3f}\n\n{text}",
        encoding="utf-8",
    )
    print(f"untraced={ms:.2f} ms/s  traced={ms_tm:.2f} ms/s  tm_peak={peak/(1024*1024):.2f} MB")
    stats.print_stats(15)
    print("wrote", out_p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
