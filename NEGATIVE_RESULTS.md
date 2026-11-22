# Negative results

Experiments that did **not** beat the locked floor-hug baseline on held-out checks.
Kept here so we do not re-litigate them after the 21 Nov freeze.

## Hover / `|a|≈g` dynamic gate + ZUPT / speed-cap / leak stack

Discarded. Made degraded≈clean by killing motion, lost RPE-trans and blind-5 s to stay-put
on MH02–04 (34–59 m last-5 s). Root cause: horizontal accel barely moves `|f|`, so the
gate treated real motion as hover.

## 1st-order velocity high-pass (τ = 5–10 s)

Left a constant `v ≈ b·τ` floor under residual accel bias → km-scale walk on real EuRoC.
Replaced by cascaded 2nd-order HP with shorter τ (0.35 s).

## Mid-flight Mahony / soft gravity aiding on translation path

Treated horizontal specific force as gravity → ~2g world residual. Protect-window-only tilt
aid retained for attitude init/refine.

## Blind-window constant-ω / “tiny attitude drip”

Simulation and MH checks: attitude **hold** beats constant-ω coast and damped coast on
rotational error in the last 5 s. Drip removed.

## Severity-scheduled τ (proxy)

Compared fixed τ=0.15 (stronger late HP) vs locked τ=0.35. Did **not** beat stay-put on
all 5 while improving over 0.35. Kept constant τ=0.35.

## Low-gain gated gravity aiding (attitude only)

Tried `gravity_kp=0.05` on quiet gyro samples. **Worsened** ATE-rot on all 5 sequences
(e.g. MH01 3.6°→6.0°, MH03 1.4°→1.8°) and raised RPE-rot. MH03 error is mostly tilt
(1.34°) not heading (0.52°), but mid-flight gravity aid still hurt. Kept protect-window
Mahony only.

## Full-rate gyro attitude (drop 50 Hz keyframes)

Matched sequential `att_scan` to ~1e-15, but **RPE-rot / ATE-rot gains collapsed** vs the
locked 50 Hz block-average baselines (gate 0/5). High-rate noise integration hurts the
rotation metrics the gate guards. **Kept ~50 Hz keyframes.**

## Scalar IMU precompute under tracemalloc

Per-sample Python scans to avoid `ndarray.mean` on memmap slices: traced ms/s got
**worse** (~102 → ~160). Tracemalloc tax on millions of Python floats exceeds NumPy
means on tiny blocks.

## What cleared the traced budget (kept)

Float32 cumsum block means + protect-only Mahony + `fastpath.att_scan` / `hp_block` +
`tum_bytes` writer (no `savetxt`). Traced ~16 ms/s, heap ~3.5 MB, gate 15/15.
