# Track 03 — Path Finder

6-DoF trajectory from **degraded IMU only** (Samsung XR Hackathon 2026).

## Architecture

```text
main.py
  └─ estimator.run_estimator          dual-pass (untraced time + tracemalloc heap)
       ├─ io_utils.load_imu_csv       chunked loadtxt → float32 memmap sensors
       ├─ quality.analyze_imu         spikes / stuck / gaps / blind window
       ├─ fast_ins.run_lean_ins
       │    ├─ estimate_biases_and_q0   time-horizon bias + gravity align
       │    ├─ protect Mahony (~2 s)    attitude quality only
       │    ├─ fastpath.att_scan        vectorized post-protect gyro attitude
       │    └─ fastpath.hp_block        cascaded 2nd-order velocity HP (τ=0.35 s)
       └─ write_tum_from_keyframes    upsample + tum_bytes (no savetxt)
```

Numpy-only in the measured path. No scipy. All imports at module top.

## Approach

- Per-file gyro/accel bias + protect-window gravity alignment (no EuRoC hard-codes)
- Attitude + HP at ~50 Hz keyframes (rate from median `dt`), upsampled to every row
- Always integrate translation; **cascaded 2nd-order velocity high-pass** hugs the stay-put floor
- Blind last ~5 s: **hold** position and attitude
- Exception-safe: any failure still writes a valid hold-pose trajectory (1 row / input row)

## Quick start

```bash
pip install -r requirements.txt
# optional for local scoring: pip install evo matplotlib pandas

python src/main.py path/to/degraded_imu.csv --out estimated_trajectory.txt --meta metadata.json
python tools/validate_output.py path/to/degraded_imu.csv estimated_trajectory.txt
```

`main.py` also accepts a **folder** with `imu0/data.csv` (or `mav0/imu0/data.csv`) or a **zip** of that layout.

### One-command bench

```bash
python tools/run_synthetic_benchmark.py
python tools/regression_gate.py --seeds 42,7,99
```

### Docker

```bash
docker build -t track3-pathfinder .
docker run --rm track3-pathfinder
```

## Constraints

| Budget | Ours (MH01 degraded) |
|---|---|
| ≤8 MB algorithm heap | **tracemalloc ~3.5 MB** |
| ≤20 ms / 1 s IMU | **~2.4 ms/s** untraced; **~16 ms/s** traced |
| 1 pose per IMU row | validated |
| Python-only `main.py` | no scipy |

### Measurement protocol (also in `metadata.json`)

Two separate passes after `import numpy`:

1. **Runtime pass** — no tracemalloc → `ms_per_imu_second`, `rss_delta_*`
2. **Heap pass** — tracemalloc on → `tracemalloc_peak_*`, `ms_per_imu_second_traced`

| Field | Role |
|---|---|
| `tracemalloc_peak_mb` | **Headline** (FAQ-like buffer profile) |
| `peak_algorithm_mb` | Supporting analytical account |
| `rss_delta_mb` | Supporting RSS growth after imports |
| `ms_per_imu_second_traced` | Exposure if grader times a traced run |
| `dropout_row_frac` / `dropout_time_frac` | Gap / missing-time dropout metrics |
| `measurement_protocol` | Machine-readable copy of this section |

Spill: system temp memmap (never CWD); blocked temp → RAM. `TRACK3_TRACEMALLOC=0` skips heap pass.

## Results

See [`results.md`](results.md), [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md). Regression gate **15/15**.

## Submission zip

```bash
python tools/pack_submission.py --team "SRMIST KTR_2 fast 2 curious" \
  --traj outputs/MH_01_easy/estimated_trajectory.txt \
  --meta outputs/MH_01_easy/metadata.json
```

```text
SRMIST KTR_2 fast 2 curious.zip
├── estimated_trajectory.txt
├── metadata.json
├── README.md
└── src/
    ├── main.py
    ├── estimator.py
    ├── fast_ins.py
    ├── fastpath.py
    ├── quality.py
    ├── io_utils.py
    ├── mem_utils.py
    └── README.md
```

Team: **SRMIST KTR_2 fast 2 curious** (Track 03 Path Finder).

**Demo video (≤3 min):** [`SRMIST_KTR_2_fast_2_curious_demo.mp4`](./SRMIST_KTR_2_fast_2_curious_demo.mp4)

## Layout

- `src/` — estimator (`main.py` entry)
- `degrade_imu.py` — official degrader (local testing only)
- `tools/` — gate, validator, robustness matrix, packer
- `EMAIL_ORGANIZERS.md` — efficiency / pose scoring clarifications
