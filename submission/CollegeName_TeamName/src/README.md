# src — Track 3 estimator

```bash
python main.py <degraded_imu.csv|folder|zip> --out estimated_trajectory.txt --meta metadata.json
```

From repo root:

```bash
python src/main.py <degraded_imu.csv> --out estimated_trajectory.txt --meta metadata.json
```

## Modules

| File | Role |
|---|---|
| `main.py` | CLI entry; hold-pose fallback on failure |
| `estimator.py` | Dual-pass orchestration + metadata |
| `fast_ins.py` | Bias init + lean INS (protect Mahony → att_scan → hp_block) |
| `fastpath.py` | Vectorized attitude scan, HP, TUM bytes |
| `quality.py` | Usable / failed / gap / dup flags |
| `io_utils.py` | Chunked CSV load, keyframe→TUM upsample |
| `mem_utils.py` | RSS, spill memmap, memory report |

## Guarantees

- Auto-detects EuRoC (`ts,gyro,accel`) vs accel-first (`ts,accel,gyro`)
- Accepts `imu0/data.csv` inside a folder or zip
- On any exception, writes a valid hold-pose trajectory (one row per input row)
- Sensors spill to **system-temp memmap** (never CWD); blocked temp → in-RAM float32
- `metadata.json` reports `tracemalloc_peak_*`, `ms_per_imu_second(_traced)`, dropout fracs
- Set `TRACK3_TRACEMALLOC=0` to skip the heap pass
