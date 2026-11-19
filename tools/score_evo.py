#!/usr/bin/env python3
"""Score estimated_trajectory.txt against EuRoC/TUM ground truth using evo."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gt", help="Ground-truth TUM trajectory")
    ap.add_argument("est", help="Estimated TUM trajectory")
    ap.add_argument("--out-dir", default="outputs/evo")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ape_zip = out / "ape.zip"
    rpe_zip = out / "rpe.zip"

    ape_cmd = [
        sys.executable,
        "-m",
        "evo.main_ape",
        "tum",
        args.gt,
        args.est,
        "-a",  # SE(3) Umeyama alignment (matches FAQ)
        "--save_results",
        str(ape_zip),
    ]
    rpe_cmd = [
        sys.executable,
        "-m",
        "evo.main_rpe",
        "tum",
        args.gt,
        args.est,
        "-a",
        "--pose_relation",
        "trans_part",
        "--delta",
        "1.0",
        "--delta_unit",
        "s",
        "--save_results",
        str(rpe_zip),
    ]
    if not args.no_plot:
        ape_cmd += ["--plot", "--plot_mode", "xyz"]

    print(">>", " ".join(ape_cmd))
    r1 = subprocess.run(ape_cmd, check=False)
    print(">>", " ".join(rpe_cmd))
    r2 = subprocess.run(rpe_cmd, check=False)

    summary = {"ape_exit": r1.returncode, "rpe_exit": r2.returncode}
    # Try to read stats from zip if present
    try:
        import zipfile

        from evo.tools import file_interface

        if ape_zip.exists():
            # evo result zip contains stats.json in newer versions; fallback parse
            with zipfile.ZipFile(ape_zip) as zf:
                names = zf.namelist()
                summary["ape_files"] = names
                for cand in ("stats.json", "ape_stats.json"):
                    if cand in names:
                        summary["ape_stats"] = json.loads(zf.read(cand))
                        break
        if rpe_zip.exists():
            with zipfile.ZipFile(rpe_zip) as zf:
                names = zf.namelist()
                summary["rpe_files"] = names
                for cand in ("stats.json", "rpe_stats.json"):
                    if cand in names:
                        summary["rpe_stats"] = json.loads(zf.read(cand))
                        break
    except Exception as e:
        summary["parse_error"] = str(e)

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if r1.returncode == 0 else r1.returncode


if __name__ == "__main__":
    raise SystemExit(main())
