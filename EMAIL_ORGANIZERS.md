# Email to organizers (send today)

**To:** prism@samsung.com  
**Subject:** Track 3 Path Finder — efficiency scoring + pose metrics clarification

---

Hello Track 3 organizers,

Two quick clarifications that affect how we finalize our submission:

1. **Efficiency (runtime / memory)** — Is the ≤20 ms per IMU-second and ≤8 MB algorithm-heap budget a **pass/fail gate**, or is efficiency **scored continuously** (lower = better) as part of the reported ~40% efficiency weight?

2. **Trajectory scoring** — Is evaluation **translation-only** after SE(3) Umeyama, or is **full pose** (including rotation / RPE-rot) part of the official score?

Context: our estimator already meets the stay-put translation floor under `degrade_imu.py`. Knowing whether efficiency is continuous and whether rotation is scored will decide whether we invest remaining time in streaming/flat memory vs. attitude presentation.

Thank you,  
[College] / [Team] / [Names]
