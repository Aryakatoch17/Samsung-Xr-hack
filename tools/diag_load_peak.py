"""Finer-grained peak during load_imu_csv only."""
from __future__ import annotations

import sys
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src import io_utils


def main() -> None:
    path = ROOT / "data" / "euroc" / "MH_01_easy" / "imu.csv"
    # warm
    io_utils.load_imu_csv(path)

    tracemalloc.start(25)
    header_cols, skip = io_utils._peek_header(path)
    print("after peek", tracemalloc.get_traced_memory()[1] / 1e6)
    n = io_utils._count_data_rows(path, skip)
    print("after count", n, tracemalloc.get_traced_memory()[1] / 1e6, "MB peak")
    from src.mem_utils import open_sensor_stores, finalize_sensor_stores, cleanup_spill

    t_s = np.empty(n, dtype=np.float64)
    print("after t_s alloc", tracemalloc.get_traced_memory()[1] / 1e6)
    gyro_w, accel_w, tmp, mode = open_sensor_stores(n)
    print("after stores", mode, tracemalloc.get_traced_memory()[1] / 1e6)
    filled = 0
    for i, (raw_ts, c1, c2) in enumerate(io_utils._iter_csv_chunks(path, skip)):
        k = len(raw_ts)
        t_s[filled : filled + k] = raw_ts
        gyro_w[filled : filled + k] = c1
        accel_w[filled : filled + k] = c2
        filled += k
        if i < 3 or i % 5 == 0:
            cur, peak = tracemalloc.get_traced_memory()
            print(f"chunk {i} filled={filled} cur={cur/1e6:.3f} peak={peak/1e6:.3f}")
    gyro, accel = finalize_sensor_stores(gyro_w, accel_w, tmp, mode)
    cur, peak = tracemalloc.get_traced_memory()
    print("done cur", cur / 1e6, "peak", peak / 1e6)
    snap = tracemalloc.take_snapshot()
    for s in snap.statistics("traceback")[:5]:
        print(f"--- live {s.size/1e6:.3f} MB")
        print("".join(s.traceback.format()[-3:]))
    cleanup_spill(tmp)
    tracemalloc.stop()


if __name__ == "__main__":
    main()
