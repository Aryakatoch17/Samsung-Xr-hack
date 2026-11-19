"""IMU CSV load/save and TUM trajectory writer."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

G = 9.80665


def _detect_ts_scale(ts: np.ndarray) -> Tuple[float, str]:
    if len(ts) < 2:
        return 1e-9, "ns"
    dt = float(np.median(np.diff(ts.astype(np.float64))))
    if dt < 1.0:
        return 1.0, "s"
    if dt < 1e3:
        return 1e-3, "ms"
    if dt < 1e6:
        return 1e-6, "us"
    return 1e-9, "ns"


def _detect_accel_scale(acc: np.ndarray) -> Tuple[float, str]:
    norm = float(np.median(np.linalg.norm(acc, axis=1)))
    if 0.5 < norm < 2.0:
        return G, "g"  # multiply by G to get m/s^2
    return 1.0, "m/s^2"


def _parse_header_and_data(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.readline().strip()
    if not first:
        raise ValueError(f"empty file: {path}")

    header_cols = None
    skip = 0
    if first.lstrip().startswith("#"):
        has_header = True
        header_cols = [c.strip().lstrip("#") for c in first.split(",")]
        skip = 1
    else:
        parts = first.split(",")
        try:
            float(parts[0])
            has_header = False
        except ValueError:
            has_header = True
            header_cols = [c.strip() for c in parts]
            skip = 1

    data = np.loadtxt(path, delimiter=",", skiprows=skip, dtype=np.float64, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 7:
        raise ValueError(f"expected >=7 columns, got {data.shape[1]} in {path}")
    return data[:, :7], header_cols, has_header


def _looks_like_gyro(col: np.ndarray) -> bool:
    """Gyro magnitudes are typically << g; accel at rest ~9.8."""
    med = float(np.median(np.linalg.norm(col, axis=1)))
    return med < 2.0  # rad/s typical; accel would be ~9.8


def load_imu_csv(path: str | Path) -> dict:
    """
    Load IMU CSV. Supports:
      - EuRoC: timestamp, w_x, w_y, w_z, a_x, a_y, a_z
      - degrade_imu style: timestamp, a_x, a_y, a_z, g_x, g_y, g_z

    Returns dict with float64 arrays:
      t_ns (int64), t_s (float64 unix seconds), gyro (N,3), accel (N,3),
      meta info for debugging.
    """
    path = Path(path)
    data, header_cols, has_header = _parse_header_and_data(path)
    ts = data[:, 0]
    c1 = data[:, 1:4]
    c2 = data[:, 4:7]

    order = "unknown"
    if header_cols and len(header_cols) >= 7:
        h = ",".join(header_cols).lower()
        if "w_rs" in h or "gyro" in h or h.find("w_") >= 0:
            # EuRoC-like header: gyro then accel
            if header_cols[1].lower().startswith("w") or "gyro" in header_cols[1].lower():
                gyro, accel, order = c1, c2, "gyro_accel"
            elif header_cols[1].lower().startswith("a") or "accel" in header_cols[1].lower():
                accel, gyro, order = c1, c2, "accel_gyro"
        if order == "unknown":
            # fall through to heuristic
            pass

    if order == "unknown":
        # Heuristic: which block has median norm near g?
        n1 = float(np.median(np.linalg.norm(c1, axis=1)))
        n2 = float(np.median(np.linalg.norm(c2, axis=1)))
        # Prefer block closer to G as accel (ignore near-zero failure tails by using median)
        if abs(n1 - G) < abs(n2 - G):
            accel, gyro, order = c1, c2, "accel_gyro"
        elif abs(n2 - G) < abs(n1 - G):
            gyro, accel, order = c1, c2, "gyro_accel"
        else:
            # both weird (e.g. heavily degraded): assume EuRoC order
            gyro, accel, order = c1, c2, "gyro_accel_assumed"

    ts_scale, ts_unit = _detect_ts_scale(ts)
    acc_scale, acc_unit = _detect_accel_scale(accel)
    accel = accel * acc_scale

    # Convert timestamps to unix seconds and integer ns for reference
    t_s = ts.astype(np.float64) * ts_scale
    if not np.isfinite(t_s).all():
        raise ValueError(f"non-finite timestamps in {path}")
    t_ns = np.rint(t_s * 1e9).astype(np.int64)

    return {
        "t_s": t_s,
        "t_ns": t_ns,
        "gyro": gyro.astype(np.float64),
        "accel": accel.astype(np.float64),
        "order": order,
        "ts_unit": ts_unit,
        "acc_unit": acc_unit,
        "path": str(path),
        "n": int(len(t_s)),
    }


def write_tum_trajectory(path: str | Path, t_s: np.ndarray, pos: np.ndarray, quat_xyzw: np.ndarray) -> None:
    """Write estimated_trajectory.txt in TUM format: t tx ty tz qx qy qz qw."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    q = quat_xyzw.astype(np.float64, copy=True)
    nrm = np.linalg.norm(q, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    q /= nrm
    out = np.column_stack([t_s, pos, q])
    with open(path, "w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        np.savetxt(f, out, fmt="%.9f")


def write_tum_from_keyframes(
    path: str | Path,
    t_s: np.ndarray,
    t_d: np.ndarray,
    pos_d: np.ndarray,
    quat_d: np.ndarray,
    failed: np.ndarray,
    dup: np.ndarray,
    chunk: int = 8192,
) -> None:
    """Stream full-rate TUM rows from decimated keys in chunks (no full Nx7)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(t_s)
    qd = quat_d.astype(np.float64, copy=True)
    for k in range(1, len(qd)):
        if np.dot(qd[k], qd[k - 1]) < 0.0:
            qd[k] *= -1.0
    fs = int(np.flatnonzero(failed)[0]) if failed.any() else n
    with open(path, "w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            ts = t_s[start:end]
            pos = np.empty((end - start, 3), dtype=np.float64)
            quat = np.empty((end - start, 4), dtype=np.float64)
            for j in range(3):
                pos[:, j] = np.interp(ts, t_d, pos_d[:, j])
            for j in range(4):
                quat[:, j] = np.interp(ts, t_d, qd[:, j])
            nrm = np.linalg.norm(quat, axis=1, keepdims=True)
            quat /= np.maximum(nrm, 1e-12)
            # Blind hold
            local_fs = max(0, fs - start)
            if local_fs < (end - start):
                if local_fs > 0:
                    pos[local_fs:] = pos[local_fs - 1]
                    quat[local_fs:] = quat[local_fs - 1]
                elif start > 0:
                    # hold value comes from previous chunk's last good — use keyframe at fs-1
                    i_hold = max(0, fs - 1)
                    p_h = np.array([np.interp(t_s[i_hold], t_d, pos_d[:, j]) for j in range(3)])
                    q_h = np.array([np.interp(t_s[i_hold], t_d, qd[:, j]) for j in range(4)])
                    q_h /= np.linalg.norm(q_h) + 1e-15
                    pos[:] = p_h
                    quat[:] = q_h
                else:
                    pos[:] = 0.0
                    quat[:] = np.array([0.0, 0.0, 0.0, 1.0])
            # Duplicates / non-positive dt inside chunk
            bad = dup[start:end].copy()
            if end - start > 1:
                bad[1:] |= np.diff(ts) <= 0.0
            if start > 0 and (dup[start] or float(t_s[start] - t_s[start - 1]) <= 0.0):
                bad[0] = True
            if bad.any():
                idxs = np.flatnonzero(bad)
                for li in idxs:
                    gi = start + int(li)
                    if gi > 0:
                        # previous row already written — re-interp prev timestamp
                        pos[li] = np.array(
                            [np.interp(t_s[gi - 1], t_d, pos_d[:, j]) for j in range(3)]
                        )
                        qprev = np.array(
                            [np.interp(t_s[gi - 1], t_d, qd[:, j]) for j in range(4)]
                        )
                        quat[li] = qprev / (np.linalg.norm(qprev) + 1e-15)
                        if gi - 1 >= fs:
                            # already held above
                            pass
            out = np.column_stack([ts, pos, quat])
            np.savetxt(f, out, fmt="%.9f")


def write_hold_pose_trajectory(path: str | Path, t_s: np.ndarray) -> None:
    """One hold pose per row. Tiny quat variation keeps Umeyama non-degenerate."""
    n = len(t_s)
    pos = np.zeros((n, 3), dtype=np.float64)
    # Identity quat with a microscopic yaw drip vs time (not constant-ω coast).
    quat = np.zeros((n, 4), dtype=np.float64)
    quat[:, 3] = 1.0
    if n > 1:
        span = max(float(t_s[-1] - t_s[0]), 1e-6)
        yaw = 1e-6 * ((t_s - t_s[0]) / span)
        half = 0.5 * yaw
        quat[:, 2] = np.sin(half)
        quat[:, 3] = np.cos(half)
    write_tum_trajectory(path, t_s, pos, quat)


def resolve_imu_path(path: str | Path) -> Path:
    """Accept a CSV, a folder containing imu0/data.csv, or a zip with that layout."""
    import zipfile
    import tempfile

    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".csv":
        return p
    if p.is_file() and p.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="track3_imu_"))
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(tmp)
        return resolve_imu_path(tmp)
    if p.is_dir():
        candidates = [
            p / "imu0" / "data.csv",
            p / "mav0" / "imu0" / "data.csv",
            p / "data.csv",
            p / "imu.csv",
        ]
        for c in candidates:
            if c.exists():
                return c
        # Any single csv in the tree
        csvs = list(p.rglob("*.csv"))
        if len(csvs) == 1:
            return csvs[0]
        for c in csvs:
            if c.name == "data.csv" and "imu" in str(c).lower():
                return c
        raise FileNotFoundError(f"no imu CSV found under {p}")
    raise FileNotFoundError(f"IMU path not found: {p}")


def euroc_to_degrade_format(in_path: str | Path, out_path: str | Path) -> None:
    """
    Convert EuRoC imu0/data.csv (ts,gx,gy,gz,ax,ay,az) to degrade_imu.py
    expected order (ts,ax,ay,az,gx,gy,gz).
    """
    imu = load_imu_csv(in_path)
    # Force interpret as EuRoC if needed
    data = np.column_stack(
        [
            imu["t_ns"] if imu["ts_unit"] == "ns" else np.rint(imu["t_s"] * 1e9),
            imu["accel"],
            imu["gyro"],
        ]
    )
    # If original was already accel-first, load_imu already swapped — use raw reload
    raw, _, has_header = _parse_header_and_data(Path(in_path))
    # Always write from detected gyro/accel in accel-first order
    out = np.column_stack([raw[:, 0], imu["accel"], imu["gyro"]])
    header = "timestamp,ax,ay,az,gx,gy,gz"
    np.savetxt(out_path, out, delimiter=",", header=header, comments="", fmt="%.6f")
