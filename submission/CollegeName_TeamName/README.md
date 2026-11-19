# Track 03 — Path Finder

6-DoF trajectory from **degraded IMU only** (Samsung XR Hackathon 2026).

## Approach

- Per-file gyro/accel bias + protect-window gravity alignment (no EuRoC hard-codes)
- Gyro attitude at ~50 Hz, upsampled to every row
- Always integrate translation; **cascaded 2nd-order velocity high-pass** (`τ=0.35` s) hugs the stay-put floor
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

| Budget | Target (ours) |
|---|---|
| ≤8 MB algorithm heap | ~1.2–2.2 MB on MH |
| ≤20 ms / 1 s IMU | ~4–6 ms/s |
| 1 pose per IMU row | validated |
| Python-only `main.py` | no scipy in estimator |

## Results

See [`results.md`](results.md), [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md), tag `v1-safe`.

## Submission zip

```bash
python tools/pack_submission.py --team CollegeName_TeamName \
  --traj outputs/MH_01_easy/estimated_trajectory.txt \
  --meta outputs/MH_01_easy/metadata.json
```

```text
CollegeName_TeamName.zip
├── estimated_trajectory.txt
├── metadata.json
├── README.md
└── src/
    ├── main.py
    └── ...
```

## Layout

- `src/` — estimator (`main.py` entry)
- `degrade_imu.py` — official degrader (local testing only)
- `tools/` — gate, validator, horizon plot, scaling, packer
