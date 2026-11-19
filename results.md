# Results — Track 3 Path Finder

**Tag:** `v1-safe` (floor-hug baseline) → lean 50 Hz stream path (current).  
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
| Runtime | ≤10 ms/s | **~5.0–5.8 ms/s** |
| Algo heap (account) | ≤5 MB | **~1.2–2.2 MB** on MH |

Official seed 42 included. See `outputs/regression_gate.json`.

## Rotation (why we can still win)

| seq | held RPE-rot | ours | held ATE-rot | ours |
|---|---:|---:|---:|---:|
| MH01 (degraded) | ~9° | **0.54°** | — | single-digit |
| MH03 | — | **0.72°** RPE-rot | tilt **1.34°** / yaw **0.52°** (clean lean) | |

Avg RPE-rot gain vs held-attitude on degraded MH: **~15–20×**.

## Efficiency

- MH01 degraded: **~3.9 ms/s**, **~1.8 MB** algorithm account  
- Scaling: linear in length; 15-min synthetic ~5 ms/s (full in-memory load ~9 MB — chunked I/O follow-up if eval files are that long)  
- Figure: `outputs/figures/scaling.png`

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
