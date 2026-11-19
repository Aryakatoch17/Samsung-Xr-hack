#!/usr/bin/env python3
"""Split MH03 tilt vs heading; try low-gain gated gravity on attitude only.

Adopt only if ATE-rot improves on ALL 5 sequences and RPE-rot does not worsen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from metrics_lib import (  # noqa: E402
    ate_rot_best_const,
    quat_angle_deg,
    quat_conj,
    quat_mul,
    quat_xyzw_from_pose,
    rpe_rot_rmse,
    pairs_1s,
)
from evo.core import sync
from evo.tools import file_interface
from src.estimator import run_estimator

SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]


def tilt_heading_errors(poses_gt, poses_est):
    """RMSE tilt (roll/pitch) and heading (yaw) after best const yaw-free? use full best const."""
    # Relative attitude error per frame after Markley align is in metrics; here decompose.
    from metrics_lib import mean_quat

    qg = [quat_xyzw_from_pose(T) for T in poses_gt]
    qe = [quat_xyzw_from_pose(T) for T in poses_est]
    qd = mean_quat([quat_mul(g, quat_conj(e)) for g, e in zip(qg, qe)])
    tilts, yaws = [], []
    for g, e in zip(qg, qe):
        err = quat_mul(g, quat_conj(quat_mul(qd, e)))  # xyzw
        # Convert to euler zyx roughly from rotmat
        x, y, z, w = err
        # roll (x), pitch (y), yaw (z)
        sinr = 2 * (w * x + y * z)
        cosr = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr, cosr)
        sinp = 2 * (w * y - z * x)
        sinp = np.clip(sinp, -1, 1)
        pitch = np.arcsin(sinp)
        siny = 2 * (w * z + x * y)
        cosy = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny, cosy)
        tilts.append(np.sqrt(roll * roll + pitch * pitch) * 180 / np.pi)
        yaws.append(abs(yaw) * 180 / np.pi)
    return float(np.sqrt(np.mean(np.square(tilts)))), float(np.sqrt(np.mean(np.square(yaws))))


def eval_rot(gt, est):
    g = file_interface.read_tum_trajectory_file(str(gt))
    e = file_interface.read_tum_trajectory_file(str(est))
    g, e = sync.associate_trajectories(g, e, max_diff=0.05)
    pairs = pairs_1s(g.timestamps)
    return {
        "rpe_rot": rpe_rot_rmse(g.poses_se3, e.poses_se3, pairs),
        "ate_rot": ate_rot_best_const(g.poses_se3, e.poses_se3),
        "tilt_rmse": tilt_heading_errors(g.poses_se3, e.poses_se3)[0],
        "heading_rmse": tilt_heading_errors(g.poses_se3, e.poses_se3)[1],
    }


def main() -> int:
    rows = []
    for seq in SEQS:
        out = ROOT / "outputs" / seq
        gt = out / "gt_tum.txt"
        imu = ROOT / "data" / "euroc" / seq / "imu.csv"
        base_p = out / "est_att_base.txt"
        aid_p = out / "est_att_aid.txt"
        run_estimator(str(imu), str(base_p), gravity_aid=False)
        run_estimator(str(imu), str(aid_p), gravity_aid=True, gravity_kp=0.05)
        b = eval_rot(gt, base_p)
        a = eval_rot(gt, aid_p)
        row = {"seq": seq, "base": b, "aid": a,
               "ate_rot_improved": a["ate_rot"] < b["ate_rot"],
               "rpe_rot_not_worse": a["rpe_rot"] <= b["rpe_rot"] * 1.001}
        rows.append(row)
        print(
            f"{seq}: base ATE-rot={b['ate_rot']:.2f} RPE-rot={b['rpe_rot']:.2f} "
            f"tilt={b['tilt_rmse']:.2f} yaw={b['heading_rmse']:.2f} | "
            f"aid ATE-rot={a['ate_rot']:.2f} RPE-rot={a['rpe_rot']:.2f} "
            f"tilt={a['tilt_rmse']:.2f} yaw={a['heading_rmse']:.2f}"
        )

    adopt = all(r["ate_rot_improved"] and r["rpe_rot_not_worse"] for r in rows)
    # MH03 split detail
    mh03 = next(r for r in rows if r["seq"] == "MH_03_medium")
    print("\nMH03 tilt vs heading (base):", mh03["base"]["tilt_rmse"], mh03["base"]["heading_rmse"])
    print("ADOPT gravity_aid:" , adopt)
    out = {"rows": rows, "adopt_gravity_aid": adopt}
    (ROOT / "outputs" / "attitude_ablation.json").write_text(json.dumps(out, indent=2))
    if not adopt:
        neg = ROOT / "NEGATIVE_RESULTS.md"
        prev = neg.read_text(encoding="utf-8") if neg.exists() else "# Negative results\n\n"
        if "gravity_aid" not in prev:
            neg.write_text(
                prev
                + "\n## Low-gain gated gravity aiding (attitude only)\n"
                + "Tried `gravity_kp=0.05` on quiet gyro samples. Did **not** improve "
                + "ATE-rot on all 5 sequences without worsening RPE-rot. Kept protect-window "
                + "Mahony only.\n",
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
