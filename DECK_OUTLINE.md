# 7-slide deck outline

**Filename:** `CollegeName_TeamName.pptx` (or PDF)

1. **Title** — Track 3 Path Finder · College · Team · tagline: *Never below the stay-put floor; win on rotation & efficiency*
2. **Problem** — Official `degrade_imu.py` severity ramp + last-5 s zeroing ⇒ late information is gone; score mixes ATE, blind-5 s, dropouts, runtime
3. **Stay-put floor** — Formula + MH01–05 floors (4.3–6.9 m); km-scale naive INS vs floor-hug
4. **Method** — Bias/q0 per file → 50 Hz attitude → always-integrate + 2nd-order vel HP (τ=0.35) → blind hold; block diagram
5. **Results** — Gate 15/15; ATE≈SP; RPE-rot ~0.5–1.2° vs ~9–14° held; **V1_01 held-out** SP 1.85 m → ATE 1.84 m; MH03 ATE-rot 1.4° (tilt 1.3° / yaw 0.5°, not the old 12°)
6. **Efficiency** — MH01 **tracemalloc ~3.5 MB**; untraced **~2–3.5 ms/s**; traced **~13–16 ms/s** (FAQ ≤20). Dual-pass: time untraced / heap traced.
7. **Ablations / submit** — NEGATIVE_RESULTS; GitHub/Docker/gate; zip layout slide 17

Speaker notes: be honest that ATE≈tie with stay-put; do not claim kilometre→centimetre translation wins.
