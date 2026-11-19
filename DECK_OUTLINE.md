# 7-slide deck outline

**Filename:** `CollegeName_TeamName.pptx` (or PDF)

1. **Title** — Track 3 Path Finder · College · Team · tagline: *Never below the stay-put floor; win on rotation & efficiency*
2. **Problem** — Official `degrade_imu.py` severity ramp + last-5 s zeroing ⇒ late information is gone; score mixes ATE, blind-5 s, dropouts, runtime
3. **Stay-put floor** — Formula + MH01–05 floors (4.3–6.9 m); km-scale naive INS vs floor-hug
4. **Method** — Bias/q0 per file → 50 Hz attitude → always-integrate + 2nd-order vel HP (τ=0.35) → blind hold; block diagram
5. **Results** — Gate 15/15; ATE≈SP; RPE-rot ~0.5–1.2° vs ~9–14° held; MH03 tilt/yaw split; runtime ~5 ms/s, ~2 MB
6. **Ablations / limits** — What we rejected (NEGATIVE_RESULTS); translation is baseline; rotation + packaging are the upside; ask Samsung if RPE is rot or trans
7. **Repro & submit** — GitHub, Docker, `python tools/regression_gate.py`, zip layout (`estimated_trajectory.txt`, `metadata.json`, `src/main.py`, README)

Speaker notes: be honest that ATE≈tie with stay-put; do not claim kilometre→centimetre translation wins.
