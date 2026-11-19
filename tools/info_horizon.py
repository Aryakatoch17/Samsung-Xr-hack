#!/usr/bin/env python3
"""Information-horizon plot + severity-scheduled tau experiment.

Adopts scheduled tau only if it beats stay-put on ALL 5 sequences (held-out style:
tune idea on MH01–03, require win on MH04–05 too). Otherwise logs NEGATIVE_RESULTS.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from evo.core import sync
from evo.tools import file_interface
from metrics_lib import stay_put_ate
from src.estimator import run_estimator, VEL_HP_TAU_S

SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]


def severity_ramp(t_frac: np.ndarray, gamma: float = 1.7, floor: float = 0.02) -> np.ndarray:
    return floor + (1.0 - floor) * np.clip(t_frac, 0, 1) ** gamma


def rolling_ate(gt_xyz, est_xyz, win_s, t, step_s=1.0):
    """Approx local RMSE in windows (no re-align per window — after global Umeyama)."""
    out_t, out_e, out_sp = [], [], []
    t0, t1 = float(t[0]), float(t[-1])
    c = gt_xyz.mean(0)
    tt = t0
    while tt + win_s <= t1 + 1e-9:
        mask = (t >= tt) & (t < tt + win_s)
        if mask.sum() >= 5:
            g = gt_xyz[mask]
            e = est_xyz[mask]
            out_t.append(tt + 0.5 * win_s)
            out_e.append(float(np.sqrt(np.mean(np.sum((g - e) ** 2, 1)))))
            out_sp.append(float(np.sqrt(np.mean(np.sum((g - c) ** 2, 1)))))
        tt += step_s
    return np.array(out_t), np.array(out_e), np.array(out_sp)


def scheduled_tau(t_frac: float, tau0: float = 0.35, tau1: float = 0.15) -> float:
    """Shorter tau (stronger HP) as severity rises — experimental."""
    sev = severity_ramp(np.array([t_frac]))[0]
    return float(tau0 + (tau1 - tau0) * sev)


def main() -> int:
    # Use MH01 for the figure; evaluate scheduled idea on all 5 via fixed end-tau proxy.
    # True per-sample scheduled tau would require estimator hooks; we proxy with two fixed
    # taus: baseline 0.35 vs aggressive 0.15 (late-file emphasis).
    fig_dir = ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    seq = "MH_01_easy"
    out = ROOT / "outputs" / seq
    gt_path = out / "gt_tum.txt"
    imu = ROOT / "data" / "euroc" / seq / "imu.csv"
    est_path = out / "est_horizon.txt"
    run_estimator(str(imu), str(est_path), vel_hp_tau_s=VEL_HP_TAU_S)

    g = file_interface.read_tum_trajectory_file(str(gt_path))
    e = file_interface.read_tum_trajectory_file(str(est_path))
    g, e = sync.associate_trajectories(g, e, max_diff=0.05)
    e.align(g, correct_scale=False)
    t = g.timestamps
    t_frac = (t - t[0]) / max(t[-1] - t[0], 1e-6)
    sev = severity_ramp(t_frac)
    tw, err, sp = rolling_ate(g.positions_xyz, e.positions_xyz, win_s=5.0, t=t, step_s=2.0)

    # Save CSV for plotting without requiring matplotlib at import time of estimator.
    csv_path = fig_dir / "info_horizon_MH01.csv"
    # Align severity on window centers
    sev_w = np.interp(tw, t, sev)
    np.savetxt(
        csv_path,
        np.column_stack([tw - t[0], err, sp, sev_w]),
        header="t_rel_s,tracking_rmse,stayput_rmse,severity",
        comments="",
        delimiter=",",
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(tw - t[0], err, label="tracking (5 s window RMSE)")
        ax1.plot(tw - t[0], sp, label="stay-put (same windows)", linestyle="--")
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("RMSE (m)")
        ax2 = ax1.twinx()
        ax2.plot(tw - t[0], sev_w, color="gray", alpha=0.5, label="severity")
        ax2.set_ylabel("severity")
        ax1.set_title("Information horizon — MH01 (severity ramp overlay)")
        ax1.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(fig_dir / "info_horizon_MH01.png", dpi=140)
        plt.close(fig)
        print("wrote", fig_dir / "info_horizon_MH01.png")
    except Exception as ex:
        print("matplotlib plot skipped:", ex)

    # Proxy scheduled-tau test: tau=0.15 vs baseline on all 5 (must beat SP everywhere).
    results = []
    adopt = True
    for s in SEQS:
        o = ROOT / "outputs" / s
        imu_s = ROOT / "data" / "euroc" / s / "imu.csv"
        gt_s = o / "gt_tum.txt"
        p_base = o / "est_tau035.txt"
        p_agg = o / "est_tau015.txt"
        run_estimator(str(imu_s), str(p_base), vel_hp_tau_s=0.35)
        run_estimator(str(imu_s), str(p_agg), vel_hp_tau_s=0.15)
        for label, p in [("tau035", p_base), ("tau015", p_agg)]:
            gg = file_interface.read_tum_trajectory_file(str(gt_s))
            ee = file_interface.read_tum_trajectory_file(str(p))
            gg, ee = sync.associate_trajectories(gg, ee, max_diff=0.05)
            ee.align(gg, correct_scale=False)
            ate = float(
                np.sqrt(np.mean(np.sum((gg.positions_xyz - ee.positions_xyz) ** 2, 1)))
            )
            spv = stay_put_ate(gg.positions_xyz)
            results.append({"seq": s, "variant": label, "ate": ate, "stayput": spv, "beats_sp": ate < spv})
        # scheduled proxy must beat SP on this seq
        row_agg = results[-1]
        if not row_agg["beats_sp"]:
            adopt = False

    # Actually require ALL seqs beat SP for tau015 AND improve avg vs tau035 — strict.
    by_seq = {}
    for r in results:
        by_seq.setdefault(r["seq"], {})[r["variant"]] = r
    adopt = True
    for s, d in by_seq.items():
        if not (d["tau015"]["beats_sp"] and d["tau015"]["ate"] <= d["tau035"]["ate"]):
            adopt = False

    print("ADOPT severity-scheduled tau (proxy tau=0.15):", adopt)
    (ROOT / "outputs" / "info_horizon.json").write_text(
        json.dumps({"results": results, "adopt_scheduled_tau": adopt}, indent=2)
    )
    if not adopt:
        neg = ROOT / "NEGATIVE_RESULTS.md"
        prev = neg.read_text(encoding="utf-8")
        if "Severity-scheduled" in prev and "proxy tau=0.15" not in prev:
            pass
        if "proxy tau=0.15" not in prev:
            neg.write_text(
                prev
                + "\n### Severity-scheduled τ (proxy)\n"
                + "Compared fixed τ=0.15 (stronger late HP) vs locked τ=0.35. "
                + "Did **not** beat stay-put on all 5 while improving over 0.35. "
                + "Kept constant τ=0.35.\n",
                encoding="utf-8",
            )
    print("wrote", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
