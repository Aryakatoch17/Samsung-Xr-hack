#!/usr/bin/env python3
"""RO-cwd smoke test + dual memory report (account / tracemalloc / RSS)."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.estimator import run_estimator  # noqa: E402


def _chmod_tree(path: Path, mode: int) -> None:
    for p in [path, *path.rglob("*")]:
        try:
            os.chmod(p, mode)
        except Exception:
            pass


def main() -> int:
    imu = ROOT / "data" / "euroc" / "MH_01_easy" / "imu.csv"
    if not imu.exists():
        # Fall back to a short synthetic
        import numpy as np

        imu = ROOT / "outputs" / "scaling" / "imu_ro_probe.csv"
        imu.parent.mkdir(parents=True, exist_ok=True)
        n = 4000
        t = 1_400_000_000.0 + np.arange(n) / 200.0
        g = np.zeros((n, 3))
        a = np.zeros((n, 3))
        a[:, 2] = 9.80665
        np.savetxt(
            imu,
            np.column_stack([np.rint(t * 1e9), g, a]),
            delimiter=",",
            header="timestamp,wx,wy,wz,ax,ay,az",
            comments="",
            fmt="%.6f",
        )

    out_root = Path(tempfile.mkdtemp(prefix="track3_ro_"))
    ro_cwd = out_root / "ro_cwd"
    writable = out_root / "writable"
    ro_cwd.mkdir()
    writable.mkdir()
    # Seed a marker so the dir isn't empty; then lock it down.
    (ro_cwd / "KEEP").write_text("x", encoding="utf-8")
    _chmod_tree(ro_cwd, stat.S_IREAD | stat.S_IEXEC)
    if sys.platform != "win32":
        os.chmod(ro_cwd, 0o555)

    prev = Path.cwd()
    try:
        os.chdir(ro_cwd)
        # Attempt a CWD write — should fail or be blocked on Unix; on Windows
        # ACL may still allow owner writes, so we also force RAM spill below.
        cwd_writable = True
        try:
            Path("should_fail.txt").write_text("nope", encoding="utf-8")
            Path("should_fail.txt").unlink(missing_ok=True)
        except Exception:
            cwd_writable = False

        out = writable / "estimated_trajectory.txt"
        meta_path = writable / "metadata.json"

        # Normal path (system-temp memmap)
        meta = run_estimator(str(imu), str(out))
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Forced no-writable-temp → RAM fallback (covers grader sandboxes that
        # block mkstemp / memmap files entirely).
        import src.mem_utils as mem_utils

        real_writable = mem_utils._writable_dir
        mem_utils._writable_dir = lambda _p: False  # type: ignore[assignment]
        try:
            out2 = writable / "estimated_trajectory_ram.txt"
            meta_ram = run_estimator(str(imu), str(out2))
        finally:
            mem_utils._writable_dir = real_writable  # type: ignore[assignment]

        summary = {
            "cwd": str(Path.cwd()),
            "cwd_writable": cwd_writable,
            "memmap_run": {
                "sensor_storage": meta.get("sensor_storage"),
                "peak_algorithm_mb": meta.get("peak_algorithm_mb"),
                "tracemalloc_peak_mb": meta.get("tracemalloc_peak_mb"),
                "rss_delta_mb": meta.get("rss_delta_mb"),
                "ms_per_imu_second": meta.get("ms_per_imu_second"),
                "n_samples": meta.get("n_samples"),
                "out_exists": out.exists(),
            },
            "ram_fallback_run": {
                "sensor_storage": meta_ram.get("sensor_storage"),
                "peak_algorithm_mb": meta_ram.get("peak_algorithm_mb"),
                "tracemalloc_peak_mb": meta_ram.get("tracemalloc_peak_mb"),
                "rss_delta_mb": meta_ram.get("rss_delta_mb"),
                "out_exists": out2.exists(),
            },
        }
        print(json.dumps(summary, indent=2))
        (writable / "ro_cwd_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        ok = (
            out.exists()
            and out2.exists()
            and meta_ram.get("sensor_storage") == "ram"
            and meta.get("sensor_storage") in ("memmap", "ram")
        )
        return 0 if ok else 1
    finally:
        os.chdir(prev)
        # Unlock for cleanup
        _chmod_tree(ro_cwd, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        if sys.platform != "win32":
            try:
                os.chmod(ro_cwd, 0o755)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
