# Track 03 — Path Finder

6-DoF trajectory estimation from degraded IMU telemetry (Samsung XR Hackathon 2026).

## Approach

Offline **strapdown INS + complementary attitude/bias observer** with:

- Gravity / gyro-bias init from the protected first ~2 s
- Severity-aware gains matched to progressive jamming
- Spike / stuck / duplicate gating
- Gap coasting across missing rows
- Zero-tail detection for the last ~5 s sensor failure + constrained dead reckoning
- Second-pass re-integration with smoothed biases (smoother-like benefit, low memory)
- Tuned to stay under **8 MB algorithm RAM** and **20 ms / 1 s IMU**

## Quick start (synthetic benchmark)

```bash
python tools/run_synthetic_benchmark.py
```

Latest local numbers:
- Clean ATE RMSE ≈ **0.003 m**
- Degraded ATE RMSE ≈ **2.7 m** (organizer `degrade_imu.py`)
- Runtime ≈ **11 ms / 1 s IMU**, algo mem ≈ **2.3 MB**

## Real EuRoC

ETH zip downloads often time out from some networks. Place files manually:

```text
data/euroc/MH_01_easy/imu.csv   # mav0/imu0/data.csv
data/euroc/MH_01_easy/gt.csv    # mav0/state_groundtruth_estimate0/data.csv
```

Then:

```bash
python tools/euroc_gt_to_tum.py data/euroc/MH_01_easy/gt.csv data/euroc/MH_01_easy/gt_tum.txt
python tools/euroc_to_degrade_input.py data/euroc/MH_01_easy/imu.csv data/euroc/MH_01_easy/imu_accel_first.csv
python degrade_imu.py data/euroc/MH_01_easy/imu_accel_first.csv data/euroc/MH_01_easy/imu_degraded.csv
python src/main.py data/euroc/MH_01_easy/imu_degraded.csv --out outputs/euroc/estimated_trajectory.txt --meta outputs/euroc/metadata.json
python tools/score_evo.py data/euroc/MH_01_easy/gt_tum.txt outputs/euroc/estimated_trajectory.txt --out-dir outputs/euroc/evo --no-plot
```

`main.py` auto-detects EuRoC vs accel-first column order.

## Submission layout

```text
team_name.zip
├── estimated_trajectory.txt
├── metadata.json
└── src/
    ├── main.py
    └── README.md
```

## Constraints (FAQ)

- Python only (`main.py`)
- ≤ 8 MB algorithm heap (not process RSS)
- ≤ 20 ms compute per 1 s of IMU
- Exactly one output pose per input IMU row
- Offline / non-causal smoothing allowed

## Local benchmark (synthetic + degrade_imu.py)

```bash
python tools/run_synthetic_benchmark.py
```

Latest numbers:

| Split | ATE RMSE (SE3 aligned) | Notes |
|---|---|---|
| Clean IMU | ~0.003 m | sanity check |
| Degraded IMU | ~2.5 m | noise/bias/spikes/gaps + 5 s failure |
| Runtime | ~10 ms / 1 s IMU | budget 20 |
| Algo memory | ~2.3 MB | budget 8 |

### Submission pack

```bash
python tools/pack_submission.py --team CollegeName_TeamName \
  --traj outputs/synthetic/estimated_trajectory.txt \
  --meta outputs/synthetic/metadata.json
```

### Real EuRoC (when files are local)

ETH download currently times out here. Put files at:

```text
data/euroc/MH_01_easy/imu.csv   # from mav0/imu0/data.csv
data/euroc/MH_01_easy/gt.csv    # from mav0/state_groundtruth_estimate0/data.csv
```

Then:

```bash
python tools/run_euroc_local.py --seq-dir data/euroc/MH_01_easy
```
