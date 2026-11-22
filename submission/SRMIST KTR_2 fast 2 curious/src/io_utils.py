"""IMU CSV load/save and TUM trajectory writer."""

from __future__ import annotations

import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np

from .fastpath import tum_bytes
from .mem_utils import cleanup_spill, finalize_sensor_stores, open_sensor_stores

G = 9.80665
_CHUNK_ROWS = 2048  # keep load peak << full-file size (~5 MB CSV)


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
    norm = float(np.median(np.linalg.norm(acc.astype(np.float64), axis=1)))
    if 0.5 < norm < 2.0:
        return G, "g"  # multiply by G to get m/s^2
    return 1.0, "m/s^2"


def _peek_header(path: Path) -> tuple[list[str] | None, int]:
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.readline().strip()
    if not first:
        raise ValueError(f"empty file: {path}")
    header_cols = None
    skip = 0
    if first.lstrip().startswith("#"):
        header_cols = [c.strip().lstrip("#") for c in first.split(",")]
        skip = 1
    else:
        parts = first.split(",")
        try:
            float(parts[0])
        except ValueError:
            header_cols = [c.strip() for c in parts]
            skip = 1
    return header_cols, skip


def _count_data_rows(path: Path, skip: int) -> int:
    """Binary newline count (IMU CSVs are dense; header = ``skip`` lines)."""
    with open(path, "rb") as f:
        n_nl = sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))
        f.seek(0, 2)
        if f.tell() > 0:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                n_nl += 1
    return max(0, int(n_nl) - int(skip))


def _order_from_header(header_cols: list[str] | None) -> str:
    if not header_cols or len(header_cols) < 7:
        return "unknown"
    c1 = header_cols[1].lower()
    if c1.startswith("w") or "gyro" in c1:
        return "gyro_accel"
    if c1.startswith("a") or "accel" in c1:
        return "accel_gyro"
    h = ",".join(header_cols).lower()
    if "w_rs" in h or "gyro" in h:
        return "gyro_accel"
    return "unknown"


def _order_from_sample(c1: np.ndarray, c2: np.ndarray) -> str:
    n1 = float(np.median(np.linalg.norm(c1.astype(np.float64), axis=1)))
    n2 = float(np.median(np.linalg.norm(c2.astype(np.float64), axis=1)))
    if abs(n1 - G) < abs(n2 - G):
        return "accel_gyro"
    if abs(n2 - G) < abs(n1 - G):
        return "gyro_accel"
    return "gyro_accel_assumed"


def _parse_header_and_data(path: Path):
    """Legacy full-file float64 load (tools only). Prefer ``load_imu_csv``."""
    header_cols, skip = _peek_header(path)
    data = np.loadtxt(path, delimiter=",", skiprows=skip, dtype=np.float64, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 7:
        raise ValueError(f"expected >=7 columns, got {data.shape[1]} in {path}")
    return data[:, :7], header_cols, skip > 0


def _iter_csv_chunks(path: Path, skip: int, chunk_rows: int = _CHUNK_ROWS):
    """Yield (ts f64, c1 f32, c2 f32) via ``np.loadtxt`` on the open file handle.

    Never joins CSV lines into a giant Python string (that alone peaked ~file size).
    Peak per chunk ≈ ``chunk_rows × 7 × 8`` bytes (~0.11 MB at 2048).
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        for _ in range(skip):
            f.readline()
        while True:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                try:
                    raw = np.loadtxt(
                        f,
                        delimiter=",",
                        dtype=np.float64,
                        comments="#",
                        max_rows=chunk_rows,
                    )
                except ValueError:
                    break
            if not isinstance(raw, np.ndarray) or raw.size == 0:
                break
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            if raw.shape[1] < 7:
                raise ValueError(f"expected >=7 columns, got {raw.shape[1]} in {path}")
            yield (
                np.array(raw[:, 0], dtype=np.float64, copy=True),
                np.array(raw[:, 1:4], dtype=np.float32, copy=True),
                np.array(raw[:, 4:7], dtype=np.float32, copy=True),
            )
            del raw


def load_imu_csv(path: str | Path) -> dict:
    """
    Load IMU CSV with **no full-array float64 transient**.

    Streams chunks into float32 gyro/accel stores (system-temp memmap when
    possible, else RAM) and float64 timestamps only.

    Supports:
      - EuRoC: timestamp, w_x, w_y, w_z, a_x, a_y, a_z
      - degrade_imu style: timestamp, a_x, a_y, a_z, g_x, g_y, g_z
    """
    path = Path(path)
    header_cols, skip = _peek_header(path)
    n = _count_data_rows(path, skip)
    if n < 1:
        raise ValueError(f"no IMU rows in {path}")

    t_s = np.empty(n, dtype=np.float64)
    gyro_w, accel_w, tmp_dir, storage = open_sensor_stores(n)

    order = _order_from_header(header_cols)
    ts_scale: float | None = None
    ts_unit = "ns"
    acc_scale = 1.0
    acc_unit = "m/s^2"
    filled = 0
    probe_c1: np.ndarray | None = None
    probe_c2: np.ndarray | None = None

    for raw_ts, c1, c2 in _iter_csv_chunks(path, skip):
        k = int(raw_ts.shape[0])
        sl = slice(filled, filled + k)

        if filled == 0:
            probe_c1 = c1[: min(k, 4096)].copy()
            probe_c2 = c2[: min(k, 4096)].copy()
            if order == "unknown":
                order = _order_from_sample(probe_c1, probe_c2)
            ts_scale, ts_unit = _detect_ts_scale(raw_ts)
            if order.startswith("accel"):
                acc_scale, acc_unit = _detect_accel_scale(probe_c1)
            else:
                acc_scale, acc_unit = _detect_accel_scale(probe_c2)

        assert ts_scale is not None
        t_s[sl] = raw_ts * ts_scale

        if order.startswith("accel"):
            if acc_scale != 1.0:
                accel_w[sl] = c1 * np.float32(acc_scale)
            else:
                accel_w[sl] = c1
            gyro_w[sl] = c2
        else:
            gyro_w[sl] = c1
            if acc_scale != 1.0:
                accel_w[sl] = c2 * np.float32(acc_scale)
            else:
                accel_w[sl] = c2

        del raw_ts, c1, c2
        filled += k

    if filled != n:
        # Recount mismatch (blank lines etc.) — trim
        n = filled
        t_s = t_s[:n]
        gyro_w = gyro_w[:n]
        accel_w = accel_w[:n]

    if not np.isfinite(t_s).all():
        cleanup_spill(tmp_dir)
        raise ValueError(f"non-finite timestamps in {path}")

    gyro, accel = finalize_sensor_stores(gyro_w, accel_w, tmp_dir, storage)

    return {
        "t_s": t_s,
        # t_ns omitted on purpose — allocate only if a caller needs it (avoids +8 B/row).
        "gyro": gyro,
        "accel": accel,
        "order": order,
        "ts_unit": ts_unit,
        "acc_unit": acc_unit,
        "path": str(path),
        "n": int(n),
        "sensor_storage": storage,
        "sensor_tmp": tmp_dir,
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
    chunk: int = 2048,
) -> None:
    """Stream full-rate TUM rows from decimated keys in chunks (no full Nx7)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(t_s)
    # Hemisphere-fix on float32 keys in-place.
    qd = np.array(quat_d, dtype=np.float32, copy=True)
    for k in range(1, len(qd)):
        if float(np.dot(qd[k], qd[k - 1])) < 0.0:
            qd[k] *= -1.0
    fs = int(np.flatnonzero(failed)[0]) if failed.any() else n
    with open(path, "wb") as f:
        f.write(b"# timestamp tx ty tz qx qy qz qw\n")
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            ts = t_s[start:end]
            pos = np.empty((end - start, 3), dtype=np.float64)
            quat = np.empty((end - start, 4), dtype=np.float64)
            for j in range(3):
                pos[:, j] = np.interp(ts, t_d, pos_d[:, j])
            for j in range(4):
                quat[:, j] = np.interp(ts, t_d, qd[:, j].astype(np.float64, copy=False))
            nrm = np.linalg.norm(quat, axis=1, keepdims=True)
            quat /= np.maximum(nrm, 1e-12)
            # Blind hold
            local_fs = max(0, fs - start)
            if local_fs < (end - start):
                if local_fs > 0:
                    pos[local_fs:] = pos[local_fs - 1]
                    quat[local_fs:] = quat[local_fs - 1]
                elif start > 0:
                    i_hold = max(0, fs - 1)
                    p_h = np.array([np.interp(t_s[i_hold], t_d, pos_d[:, j]) for j in range(3)])
                    q_h = np.array([np.interp(t_s[i_hold], t_d, qd[:, j]) for j in range(4)])
                    q_h /= np.linalg.norm(q_h) + 1e-15
                    pos[:] = p_h
                    quat[:] = q_h
                else:
                    pos[:] = 0.0
                    quat[:] = np.array([0.0, 0.0, 0.0, 1.0])
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
                        pos[li] = np.array(
                            [np.interp(t_s[gi - 1], t_d, pos_d[:, j]) for j in range(3)]
                        )
                        qprev = np.array(
                            [np.interp(t_s[gi - 1], t_d, qd[:, j]) for j in range(4)]
                        )
                        quat[li] = qprev / (np.linalg.norm(qprev) + 1e-15)
            f.write(tum_bytes(ts, pos, quat))


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
    """Accept a CSV, a folder containing imu0/data.csv, or a zip with that layout.

    Zip extract goes to the system temp directory (never CWD) so a read-only
    working directory still works.
    """
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".csv":
        return p
    if p.is_file() and p.suffix.lower() == ".zip":
        # Explicitly use gettempdir — do not extract into CWD.
        tmp = Path(tempfile.mkdtemp(prefix="track3_imu_", dir=tempfile.gettempdir()))
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
    t_ns = np.rint(imu["t_s"] * 1e9)
    out = np.column_stack([t_ns, imu["accel"], imu["gyro"]])
    header = "timestamp,ax,ay,az,gx,gy,gz"
    np.savetxt(out_path, out, delimiter=",", header=header, comments="", fmt="%.6f")
    cleanup_spill(imu.get("sensor_tmp"))
