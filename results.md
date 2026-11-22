# Results — Track 3 Path Finder

**Tag:** `v1-safe` → lean 50 Hz + vectorized fastpath (current).  
**Freeze:** 21 Nov 12:00 IST · **Submit:** morning of 22 Nov.

## Design (one paragraph)

Per-file bias + protect-window gravity alignment; always integrate; cascaded 2nd-order
velocity high-pass (`τ=0.35` s) hugs the stay-put translation floor; blind last ~5 s
**holds** position and attitude; gyro-integrated attitude at ~50 Hz with upsample.
No EuRoC hard-coded biases. No scipy in `main.py`.

## Regression gate (3 degrade seeds × 5 MH sequences) — **15/15 PASS**

| Metric | Gate | Observed |
|---|---|---|
| ATE vs stay-put | ≤1.05× | all pass (typically beat SP slightly) |
| RPE-trans 1 s | ≤1.05× | all pass |
| Last-5 s | ≤1.10× | all pass |
| Rot gain vs degraded baseline | ≥90% | all pass (~7–25× vs held-attitude) |
| Runtime | ≤10 ms/s gate / ≤20 FAQ | **~2.7 ms/s** untraced; traced **~16 ms/s** |
| Algo heap (tracemalloc headline) | ≤8 MB FAQ | **~3.5 MB** MH01 |

Official seed 42 included. See `outputs/regression_gate.json`.

## Rotation (why we can still win)

**MH03 clarification:** the old **12° ATE-rot** was the *pre-lean* stack. Current lean path on clean MH03 is **ATE-rot = 1.44°** (geodesic RMSE after best constant quat align). Roll/pitch (“tilt”) RMSE **1.34°** and yaw **0.52°** are Euler components of the *same* error quaternions — they do **not** add to 12° (or even to 1.44°). `tot_rmse` of the geodesic equals ATE-rot. Degraded MH03: ATE-rot **2.12°**, tilt **1.92°**, yaw **0.92°**, RPE-rot **0.72°**. Gravity aiding still rejected.

| seq | held RPE-rot | ours RPE-rot | ATE-rot (const-align) |
|---|---:|---:|---:|
| MH01 degraded | ~9° | **0.54°** | **5.5°** |
| MH03 clean / degr | 14.3° / — | **0.15° / 0.72°** | **1.44° / 2.12°** |

## Held-out (not MH)

| seq | stay-put | ATE degr | ratio | RPE× | B5× | within targets |
|---|---:|---:|---:|---:|---:|:---:|
| V1_01_easy | 1.85 m | **1.84 m** | 0.994 | 1.04 | 0.99 | yes |
| V2_01_easy | 2.29 m | **2.26 m** | 0.985 | 1.04 | 0.97 | yes |

ETH unreachable; IMU+GT from HuggingFace `vicon_room1/2.zip`. TUM-VI not pulled (host/time).

## Dropout (10% scoring item)

Reported in `metadata.json` on every run (MH01 degraded seed=42 example):

| Field | Value | Meaning |
|---|---:|---|
| `dropout_row_frac` | **~0.00076** | Fraction of rows flagged as gaps |
| `dropout_time_frac` | **~0.032** | Missing time vs nominal `med_dt` span |
| `n_gaps` | 27 | Gap-flagged samples |

## Efficiency

- **Headline = tracemalloc** (after imports). MH01 **~2.1 MB**. FAQ ≤8 MB.  
- **Runtime:** untraced **~4.5–7 ms/s**. Traced dual-pass exposure **~60–70 ms/s** — confirm with organizers whether graders time a traced run (`EMAIL_ORGANIZERS.md`).  
- Hot path: locked 50 Hz block-average lean + `tum_bytes` writer; 30 s bias horizon.  
- Held-out: V1 within stay-put gates; V2 ATE/B5 pass, RPE-trans **1.06×** (soft).  
- No scipy. Sensor spill: system temp (never CWD).

## Negative results (do not re-open after freeze)

See `NEGATIVE_RESULTS.md`: hover gate, 1st-order HP, mid-flight Mahony on translation,
attitude drip, severity-scheduled τ, mid-flight gravity aid on attitude.

## Figures

- `outputs/figures/info_horizon_MH01.png` — tracking vs stay-put vs severity ramp  
- `outputs/figures/scaling.png` — runtime & memory vs duration  

## How to reproduce

```bash
python tools/regression_gate.py --seeds 42,7,99
python tools/validate_output.py outputs/MH_01_easy/imu_degraded.csv outputs/MH_01_easy/estimated_trajectory.txt
python tools/run_synthetic_benchmark.py
python tools/pack_submission.py --team CollegeName_TeamName --traj outputs/MH_01_easy/estimated_trajectory.txt --meta outputs/MH_01_easy/metadata.json
```
