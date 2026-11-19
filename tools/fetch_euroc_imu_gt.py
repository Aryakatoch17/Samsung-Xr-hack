#!/usr/bin/env python3
"""
Download one EuRoC sequence zip and extract only IMU + ground-truth CSVs.

Default: MH_01_easy (machine hall).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

SEQUENCES = {
    "MH_01_easy": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip",
    "MH_02_easy": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_02_easy/MH_02_easy.zip",
    "V1_01_easy": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip",
}

NEEDED_SUFFIXES = (
    "mav0/imu0/data.csv",
    "mav0/state_groundtruth_estimate0/data.csv",
)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"zip already present: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return
    print(f"downloading {url}")
    print(f"  -> {dest}")

    def hook(block, block_size, total):
        if total <= 0:
            return
        done = block * block_size
        pct = min(100.0, 100.0 * done / total)
        if block % 50 == 0:
            print(f"\r  {pct:5.1f}%  {done/1e6:.1f}/{total/1e6:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print("\n download complete")


def extract_needed(zip_path: Path, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    found = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for suf in NEEDED_SUFFIXES:
            matches = [n for n in names if n.replace("\\", "/").endswith(suf)]
            if not matches:
                print(f"WARNING: missing {suf}")
                continue
            src = matches[0]
            # Flatten into out_dir/imu0_data.csv etc.
            label = "imu.csv" if "imu0" in suf else "gt.csv"
            dest = out_dir / label
            print(f"extract {src} -> {dest}")
            with zf.open(src) as src_f, open(dest, "wb") as dst_f:
                dst_f.write(src_f.read())
            found[label] = dest
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="MH_01_easy", choices=sorted(SEQUENCES))
    ap.add_argument("--data-dir", default="data/euroc")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_dir)
    seq_dir = root / args.seq
    zip_path = root / f"{args.seq}.zip"

    if not args.skip_download:
        try:
            download(SEQUENCES[args.seq], zip_path)
        except Exception as e:
            print(f"download failed: {e}", file=sys.stderr)
            return 1

    if not zip_path.exists():
        print(f"missing zip: {zip_path}", file=sys.stderr)
        return 1

    found = extract_needed(zip_path, seq_dir)
    if "imu.csv" not in found or "gt.csv" not in found:
        print("failed to extract imu/gt", file=sys.stderr)
        return 1
    print("ready:")
    for k, v in found.items():
        print(f"  {k}: {v} ({v.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
