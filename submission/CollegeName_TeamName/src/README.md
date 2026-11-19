# Path Finder (Track 03) — Source

## Run

```bash
python main.py <degraded_imu.csv> --out estimated_trajectory.txt --meta metadata.json
```

From repo root:

```bash
python src/main.py <degraded_imu.csv> --out estimated_trajectory.txt --meta metadata.json
```

## Method

1. Load IMU (EuRoC `gyro,accel` or accel-first); auto-detect units/column order  
2. Flag duplicates, stuck samples, spikes, timestamp gaps, and the zeroed failure tail  
3. Pass 1: strapdown + complementary gravity correction + bias tracking  
4. Smooth biases offline; re-integrate (pass 2)  
5. Dead-reckon across the failure window with a robust pre-failure motion prior  
6. Write TUM `estimated_trajectory.txt` — one pose per input row  

## Limits

- Algorithm heap target ≤ 8 MB  
- Runtime target ≤ 20 ms per 1 s of IMU  
- Python only, single-thread, offline  
