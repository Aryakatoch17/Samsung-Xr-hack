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
