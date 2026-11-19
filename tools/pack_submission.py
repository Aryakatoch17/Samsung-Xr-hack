#!/usr/bin/env python3
"""
Build Track-3 submission zip:

  team_name.zip
  ├── estimated_trajectory.txt
  ├── metadata.json
  └── src/
      ├── main.py
      ├── ...
      └── README.md
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_FILES = [
    "main.py",
    "estimator.py",
    "fast_ins.py",
    "quality.py",
    "io_utils.py",
    "math_utils.py",
    "ins_ekf.py",
    "__init__.py",
    "README.md",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="CollegeName_TeamName")
    ap.add_argument("--traj", default="outputs/synthetic/estimated_trajectory.txt")
    ap.add_argument("--meta", default="outputs/synthetic/metadata.json")
    ap.add_argument("--out-dir", default="submission")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir / args.team
    if out_dir.exists():
        shutil.rmtree(out_dir)
    src_out = out_dir / "src"
    src_out.mkdir(parents=True)

    traj = ROOT / args.traj
    meta = ROOT / args.meta
    if not traj.exists():
        raise SystemExit(f"missing trajectory: {traj}")
    shutil.copy2(traj, out_dir / "estimated_trajectory.txt")
    if meta.exists():
        shutil.copy2(meta, out_dir / "metadata.json")
    else:
        (out_dir / "metadata.json").write_text("{}", encoding="utf-8")

    for name in SRC_FILES:
        p = ROOT / "src" / name
        if p.exists():
            shutil.copy2(p, src_out / name)

    # Root README for judges
    readme = ROOT / "README.md"
    if readme.exists():
        shutil.copy2(readme, out_dir / "README.md")

    zip_path = ROOT / args.out_dir / f"{args.team}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(out_dir).as_posix())

    print(f"wrote {zip_path}")
    print(f"folder {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
