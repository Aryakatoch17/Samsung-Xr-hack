# 3-minute video script (≤180 s)

**Title:** Track 3 Path Finder — hug the floor, win on rotation  
**On-screen:** CollegeName_TeamName · Samsung XR Hackathon 2026

| t | Visual | Spoken |
|---|---|---|
| 0:00–0:20 | Title + EuRoC MH01 path | “Track 3 asks for 6-DoF from a *degraded* IMU. The official script ramps severity and zeros the last five seconds — late-file information is gone.” |
| 0:20–0:45 | Stay-put vs km-scale naive INS | “We measured the stay-put floor on real ground truth. A naive integrator goes kilometres. Anything that falls below the floor is losing to doing nothing.” |
| 0:45–1:15 | Architecture diagram | “Our design: per-file bias, protect-window tilt, always integrate, then a second-order velocity high-pass so translation hugs the floor. Blind five seconds: hold pose. Attitude stays gyro-integrated.” |
| 1:15–1:45 | Table: ATE≈SP, RPE-rot 0.5° vs 9° | “On Machine Hall, translation ties stay-put within five percent. Rotation is where the IMU wins — about half a degree RPE-rot versus nine degrees for a held attitude.” |
| 1:45–2:15 | Runtime / memory bars | “Lean fifty-hertz attitude loop, streamed write: about four milliseconds per IMU-second and under two megabytes on MH01 — inside the twenty ms and eight MB budgets.” |
| 2:15–2:40 | Ablation / negative slide | “We tried hover gates and mid-flight gravity aiding — they destroyed motion or attitude. Those stay in our negative results log.” |
| 2:40–3:00 | Repo + zip layout | “Reproducible: GitHub, Docker, one-command bench, validator. Submission zip matches slide 17. Thanks.” |

**Record tips:** 1080p, burn-in captions for numbers, no background music louder than voice.
