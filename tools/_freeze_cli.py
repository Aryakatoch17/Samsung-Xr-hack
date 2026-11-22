#!/usr/bin/env python3
"""Pack CLI + fallback + zip integrity for freeze."""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.estimator import run_hold_pose_fallback
from src.io_utils import load_imu_csv
from tools.validate_output import validate


def main() -> int:
    imu = ROOT / "outputs" / "MH_01_easy" / "imu_degraded.csv"
    out_dir = ROOT / "outputs" / "freeze_cli"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj = out_dir / "estimated_trajectory.txt"
    meta = out_dir / "metadata.json"

    cmd = [
        sys.executable,
        str(ROOT / "submission" / "CollegeName_TeamName" / "src" / "main.py"),
        str(imu),
        "--out",
        str(traj),
        "--meta",
        str(meta),
    ]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

    errs = validate(imu, traj)
    print("validate packed CLI:", errs or "OK")

    import numpy as np

    hold = out_dir / "hold_fallback.txt"
    hmeta = run_hold_pose_fallback(str(imu), str(hold), err="forced")
    n = load_imu_csv(str(imu))["n"]
    rows = int(np.loadtxt(hold, comments="#").reshape(-1, 8).shape[0])
    print(
        f"fallback rows={rows} imu_n={n} fallback={hmeta.get('fallback')} match={rows == n}"
    )

    zpath = ROOT / "submission" / "CollegeName_TeamName.zip"
    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
        req = [
            "estimated_trajectory.txt",
            "metadata.json",
            "src/main.py",
            "README.md",
            "src/README.md",
        ]
        miss = [r for r in req if r not in names]
        print("zip missing:", miss or "none")
        payload = json.loads(zf.read("metadata.json"))
        blob = json.dumps(payload)
        leak = ("C:\\Users" in blob) or ("/Users/" in blob) or (":\\\\Users" in blob)
        print("meta.output=", payload.get("output"), "path_leak=", leak)
        forbidden = [n for n in names if "ins_ekf" in n or "math_utils" in n]
        print("forbidden in zip:", forbidden or "none")

    ok = (not errs) and rows == n and not miss and not leak and not forbidden
    print("RESULT:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
