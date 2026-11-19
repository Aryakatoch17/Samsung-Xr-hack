"""Shared ATE / RPE / rotation metrics vs stay-put and held-attitude baselines."""

from __future__ import annotations

import numpy as np
from evo.core import lie_algebra as lie
from evo.core import metrics, sync
from evo.tools import file_interface

RAD2DEG = 180.0 / np.pi


def stay_put_ate(xyz: np.ndarray) -> float:
    c = xyz.mean(0)
    return float(np.sqrt(np.mean(np.sum((xyz - c) ** 2, axis=1))))


def pairs_1s(t: np.ndarray, delta: float = 1.0, tol: float = 0.05):
    pairs = []
    j = 0
    for i in range(len(t)):
        tgt = t[i] + delta
        while j < len(t) and t[j] < tgt:
            j += 1
        if j >= len(t):
            break
        cands = [k for k in (j - 1, j) if 0 <= k < len(t)]
        k = min(cands, key=lambda kk: abs(t[kk] - tgt))
        if abs(t[k] - tgt) <= tol and k > i:
            pairs.append((i, k))
    return pairs


def rot_angle_deg(T: np.ndarray) -> float:
    return float(lie.so3_log_angle(T[:3, :3]) * RAD2DEG)


def rpe_trans_rmse(poses_gt, poses_est, pairs) -> float:
    errs = []
    for i, j in pairs:
        e = lie.relative_se3(
            lie.relative_se3(poses_gt[i], poses_gt[j]),
            lie.relative_se3(poses_est[i], poses_est[j]),
        )
        errs.append(np.linalg.norm(e[:3, 3]))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def rpe_rot_rmse(poses_gt, poses_est, pairs) -> float:
    errs = []
    for i, j in pairs:
        e = lie.relative_se3(
            lie.relative_se3(poses_gt[i], poses_gt[j]),
            lie.relative_se3(poses_est[i], poses_est[j]),
        )
        errs.append(rot_angle_deg(e))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def held_rpe_rot(poses_gt, pairs) -> float:
    errs = [rot_angle_deg(lie.relative_se3(poses_gt[i], poses_gt[j])) for i, j in pairs]
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def stay_rpe_trans(xyz, pairs) -> float:
    errs = [np.linalg.norm(xyz[j] - xyz[i]) for i, j in pairs]
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def quat_xyzw_from_pose(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-15
    return q


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_angle_deg(q):
    return 2.0 * np.arccos(float(np.clip(abs(q[3]), 0.0, 1.0))) * RAD2DEG


def mean_quat(qs):
    M = np.zeros((4, 4))
    for q in qs:
        q = q / (np.linalg.norm(q) + 1e-15)
        if q[3] < 0:
            q = -q
        M += np.outer(q, q)
    _, v = np.linalg.eigh(M)
    q = v[:, -1]
    if q[3] < 0:
        q = -q
    return q / np.linalg.norm(q)


def ate_rot_best_const(poses_gt, poses_est) -> float:
    qg = [quat_xyzw_from_pose(T) for T in poses_gt]
    qe = [quat_xyzw_from_pose(T) for T in poses_est]
    qd = mean_quat([quat_mul(g, quat_conj(e)) for g, e in zip(qg, qe)])
    errs = [
        quat_angle_deg(quat_mul(g, quat_conj(quat_mul(qd, e)))) for g, e in zip(qg, qe)
    ]
    return float(np.sqrt(np.mean(np.square(errs))))


def ate_rot_held(poses_gt) -> float:
    qs = [quat_xyzw_from_pose(T) for T in poses_gt]
    qm = mean_quat(qs)
    return float(np.sqrt(np.mean(np.square([quat_angle_deg(quat_mul(g, quat_conj(qm))) for g in qs]))))


def evaluate_pair(gt_path: str, est_path: str) -> dict:
    g = file_interface.read_tum_trajectory_file(str(gt_path))
    e = file_interface.read_tum_trajectory_file(str(est_path))
    g, e = sync.associate_trajectories(g, e, max_diff=0.05)
    pairs = pairs_1s(g.timestamps)

    # Rotation (no position align — Umeyama ill-conditioned for near-constant pos)
    held_rr = held_rpe_rot(g.poses_se3, pairs)
    ours_rr = rpe_rot_rmse(g.poses_se3, e.poses_se3, pairs)
    held_ar = ate_rot_held(g.poses_se3)
    ours_ar = ate_rot_best_const(g.poses_se3, e.poses_se3)

    # Translation with Umeyama
    e.align(g, correct_scale=False)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((g, e))
    ate = float(ape.get_all_statistics()["rmse"])
    rpe = rpe_trans_rmse(g.poses_se3, e.poses_se3, pairs)
    spat = stay_put_ate(g.positions_xyz)
    spr = stay_rpe_trans(g.positions_xyz, pairs)
    mask = g.timestamps >= (g.timestamps[-1] - 5.0)
    gp = g.positions_xyz[mask]
    ep = e.positions_xyz[mask]
    b5 = float(np.sqrt(np.mean(np.sum((gp - ep) ** 2, axis=1))))
    sp_b5 = float(np.sqrt(np.mean(np.sum((gp - g.positions_xyz.mean(0)) ** 2, axis=1))))

    return {
        "ate": ate,
        "rpe_1s": rpe,
        "blind5": b5,
        "stayput_ate": spat,
        "stayput_rpe": spr,
        "stayput_blind5": sp_b5,
        "rpe_rot_deg": ours_rr,
        "ate_rot_deg": ours_ar,
        "held_rpe_rot_deg": held_rr,
        "held_ate_rot_deg": held_ar,
        "rpe_rot_gain": held_rr / max(ours_rr, 1e-9),
        "ate_rot_gain": held_ar / max(ours_ar, 1e-9),
    }
