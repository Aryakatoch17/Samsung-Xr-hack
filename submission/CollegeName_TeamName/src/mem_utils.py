"""RSS / tracemalloc helpers and sensor spill (system-temp memmap → RAM fallback)."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Optional platforms — import at module load so they never land inside measured runs.
try:
    import ctypes
    from ctypes import wintypes as _wintypes
except Exception:  # pragma: no cover
    ctypes = None  # type: ignore
    _wintypes = None  # type: ignore

try:
    import psutil as _psutil
except Exception:  # pragma: no cover
    _psutil = None

try:
    import resource as _resource
except Exception:  # pragma: no cover
    _resource = None


def rss_bytes() -> int | None:
    """Current process RSS (working set). Includes memmap pages OS has faulted in."""
    if sys.platform == "win32" and ctypes is not None and _wintypes is not None:
        try:
            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", _wintypes.DWORD),
                    ("PageFaultCount", _wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS_EX()
            pmc.cb = ctypes.sizeof(pmc)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            for lib_name, fn_name in (
                ("psapi", "GetProcessMemoryInfo"),
                ("kernel32", "K32GetProcessMemoryInfo"),
            ):
                try:
                    fn = getattr(ctypes.windll, lib_name)
                    fn = getattr(fn, fn_name)
                    fn.argtypes = [
                        _wintypes.HANDLE,
                        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
                        _wintypes.DWORD,
                    ]
                    fn.restype = _wintypes.BOOL
                    if fn(handle, ctypes.byref(pmc), pmc.cb):
                        return int(pmc.WorkingSetSize)
                except Exception:
                    continue
        except Exception:
            pass
    if _psutil is not None:
        try:
            return int(_psutil.Process().memory_info().rss)
        except Exception:
            pass
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    if _resource is not None:
        try:
            usage = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return int(usage)
            return int(usage) * 1024
        except Exception:
            pass
    return None


def _writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".wprobe_{os.getpid()}"
        probe.write_bytes(b"ok")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def open_sensor_stores(n: int) -> tuple[np.ndarray, np.ndarray, str | None, str]:
    """Allocate writable float32 (n,3) gyro/accel — memmap under system temp, else RAM.

    Never writes into CWD. Caller fills stores, then ``finalize_sensor_stores``.
    """
    candidates: list[Path] = []
    try:
        candidates.append(Path(tempfile.gettempdir()) / f"track3_mm_{os.getpid()}")
    except Exception:
        pass
    for key in ("TMPDIR", "TMP", "TEMP"):
        val = os.environ.get(key)
        if val:
            candidates.append(Path(val) / f"track3_mm_{os.getpid()}")

    seen: set[str] = set()
    for base in candidates:
        key = str(base)
        if key in seen:
            continue
        seen.add(key)
        if not _writable_dir(base):
            continue
        try:
            gyro_path = str(base / "gyro.f32")
            accel_path = str(base / "accel.f32")
            gyro = np.memmap(gyro_path, dtype=np.float32, mode="w+", shape=(n, 3))
            accel = np.memmap(accel_path, dtype=np.float32, mode="w+", shape=(n, 3))
            return gyro, accel, str(base), "memmap"
        except Exception:
            try:
                shutil.rmtree(base, ignore_errors=True)
            except Exception:
                pass
            continue

    return (
        np.empty((n, 3), dtype=np.float32),
        np.empty((n, 3), dtype=np.float32),
        None,
        "ram",
    )


def finalize_sensor_stores(
    gyro: np.ndarray,
    accel: np.ndarray,
    tmp_dir: str | None,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Flush memmap writers and reopen read-only; RAM stores returned as-is."""
    if mode != "memmap" or not tmp_dir:
        return gyro, accel
    gyro.flush()
    accel.flush()
    n = int(gyro.shape[0])
    base = Path(tmp_dir)
    del gyro, accel
    gyro_r = np.memmap(str(base / "gyro.f32"), dtype=np.float32, mode="r", shape=(n, 3))
    accel_r = np.memmap(str(base / "accel.f32"), dtype=np.float32, mode="r", shape=(n, 3))
    return gyro_r, accel_r


def spill_sensors(
    gyro_src: np.ndarray,
    accel_src: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, str | None, str]:
    """Copy existing arrays into float32 stores (legacy path; prefer streamed load)."""
    gyro_w, accel_w, tmp, mode = open_sensor_stores(n)
    gyro_w[:] = np.asarray(gyro_src, dtype=np.float32)
    accel_w[:] = np.asarray(accel_src, dtype=np.float32)
    gyro, accel = finalize_sensor_stores(gyro_w, accel_w, tmp, mode)
    return gyro, accel, tmp, mode


def cleanup_spill(tmp_dir: str | None) -> None:
    if not tmp_dir:
        return
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass


def memory_report(
    *,
    account_bytes: int,
    rss_after_imports: int | None,
    rss_after: int | None,
    tracemalloc_peak: int | None,
    sensor_storage: str,
) -> dict[str, Any]:
    rss_delta = None
    if rss_after_imports is not None and rss_after is not None:
        rss_delta = max(0, int(rss_after) - int(rss_after_imports))
    out: dict[str, Any] = {
        # Headline FAQ-facing meter first; keys kept stable for tooling.
        "tracemalloc_peak_bytes": tracemalloc_peak,
        "tracemalloc_peak_mb": (
            None if tracemalloc_peak is None else tracemalloc_peak / (1024 * 1024)
        ),
        "peak_algorithm_bytes": int(account_bytes),
        "peak_algorithm_mb": account_bytes / (1024 * 1024),
        "rss_after_imports_bytes": rss_after_imports,
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": rss_delta,
        "rss_delta_mb": None if rss_delta is None else rss_delta / (1024 * 1024),
        "sensor_storage": sensor_storage,
        "measurement_protocol": {
            "heap": "tracemalloc peak on a dedicated pass started AFTER import numpy",
            "runtime": "ms_per_imu_second timed on a separate pass WITHOUT tracemalloc",
            "runtime_traced": "ms_per_imu_second_traced = wall time of the tracemalloc pass",
            "headline_meter": "tracemalloc_peak_mb",
            "supporting_meters": ["peak_algorithm_mb", "rss_delta_mb"],
        },
    }
    return out
