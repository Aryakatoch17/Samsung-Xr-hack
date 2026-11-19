# src — Track 3 estimator

```bash
python main.py <degraded_imu.csv|folder|zip> --out estimated_trajectory.txt --meta metadata.json
```

From repo root:

```bash
python src/main.py <degraded_imu.csv> --out estimated_trajectory.txt --meta metadata.json
```

- Auto-detects EuRoC (`ts,gyro,accel`) vs accel-first (`ts,accel,gyro`)
- Accepts `imu0/data.csv` inside a folder or zip
- On any exception, writes a valid hold-pose trajectory (one row per input row)
