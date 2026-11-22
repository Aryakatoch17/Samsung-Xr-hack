# 3-minute video script (≤180 s)

**Title:** Track 3 Path Finder — hug the floor, win on rotation  
**On-screen:** CollegeName_TeamName · Samsung XR Hackathon 2026  
**Replace** `CollegeName_TeamName` with your real names before upload.

| t | Visual (what to show) | Spoken |
|---|---|---|
| 0:00–0:20 | Title card + MH01 trajectory overlay | “Track 3 asks for 6-DoF from a *degraded* IMU only. The official script ramps noise and bias, drops frames, then zeros the last five seconds — late-file information is gone.” |
| 0:20–0:45 | Side-by-side: stay-put vs naive INS (km drift) | “We measured the stay-put floor on real ground truth. A naive double-integrator goes kilometres. Anything worse than the floor is losing to doing nothing.” |
| 0:45–1:20 | Architecture diagram (README block) | “Our pipeline: per-file bias and gravity align, protect-window Mahony for attitude quality, always integrate, then a second-order velocity high-pass so translation hugs the floor. Blind five seconds: hold pose. Gyro attitude keeps winning on rotation.” |
| 1:20–1:50 | Results table: ATE≈SP, RPE-rot gains | “On Machine Hall, translation stays within five percent of stay-put. Rotation is where the IMU wins — about half a degree to one degree RPE-rot versus roughly nine to fourteen degrees for a held attitude. Regression gate: fifteen of fifteen.” |
| 1:50–2:20 | Efficiency bars / metadata.json | “About two to three milliseconds per IMU-second untraced. Under tracemalloc — the FAQ-style heap — we sit around three and a half megabytes on MH01, still under eight. Traced runtime stays under the twenty millisecond budget.” |
| 2:20–2:40 | Terminal: one command run + VALID | “Live demo: degrade IMU, run main.py, validate one pose per row. Failures still write a hold-pose trajectory.” |
| 2:40–3:00 | Zip layout + GitHub + thanks | “Submission matches slide seventeen: estimated_trajectory, metadata, src/main.py. Reproducible via GitHub and Docker. Thanks.” |

**Record tips:** 1080p · burn-in captions for numbers · voice louder than any music · hard cut at ≤2:59.

## Numbers to put on screen (freeze run, seed 42)

| Metric | Value |
|---|---|
| Gate | **15/15** (3 seeds × 5 MH) |
| ATE | ≤1.05× stay-put (hugs floor) |
| RPE-rot | ~0.4–1.2° (gains ~7–25× vs held) |
| Runtime | ~2–3.5 ms/s untraced · ~13–16 ms/s traced |
| Heap | tracemalloc ~2–3.5 MB (FAQ ≤8) |

## Demo checklist — what to *show* on camera

1. **Problem beat (10 s)** — open `degrade_imu.py` / severity ramp mention; show last-5s “blind” idea.
2. **Stay-put beat** — one plot or table: stay-put ATE vs your ATE (almost equal) + naive INS callout if you have it.
3. **Architecture** — one slide/diagram only (bias → protect Mahony → 50 Hz lean → vel HP τ=0.35 → blind hold).
4. **Live terminal (must-have)**  
   ```bash
   python src/main.py outputs/MH_01_easy/imu_degraded.csv --out estimated_trajectory.txt --meta metadata.json
   python tools/validate_output.py outputs/MH_01_easy/imu_degraded.csv estimated_trajectory.txt
   ```
   Zoom on printed `ms/s`, `ms/s_traced`, `tracemalloc≈… MB`, then `VALID`.
5. **Results table** — MH01–05 ATE/SP + RPE-rot; stamp **15/15**.
6. **Negative honesty (5–10 s)** — “gravity mid-flight / hover gates failed — see NEGATIVE_RESULTS.”
7. **Submit artifact** — unzip or tree of `CollegeName_TeamName.zip` (traj + meta + `src/main.py`).

**Do not show:** debugger stepping, long scrolls of `fast_ins.py`, claims of centimetre translation, or runs with `--gravity-aid` (rejected path).
