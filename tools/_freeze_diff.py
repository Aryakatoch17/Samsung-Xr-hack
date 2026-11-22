#!/usr/bin/env python3
"""Live src vs submission pack integrity check."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "src"
PACK = ROOT / "submission" / "CollegeName_TeamName" / "src"
EXPECTED = [
    "main.py",
    "estimator.py",
    "fast_ins.py",
    "fastpath.py",
    "quality.py",
    "io_utils.py",
    "mem_utils.py",
    "__init__.py",
    "README.md",
]
FORBIDDEN = ["ins_ekf.py", "math_utils.py"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    drift = False
    print("=== LIVE vs PACK ===")
    for n in EXPECTED:
        a, b = LIVE / n, PACK / n
        if not a.exists():
            print(f"MISSING LIVE {n}")
            drift = True
            continue
        if not b.exists():
            print(f"MISSING PACK {n}")
            drift = True
            continue
        same = sha(a) == sha(b)
        if not same:
            drift = True
        tag = "OK" if same else "DRIFT"
        print(f"{tag:5} {n:16} L={a.stat().st_size:6} P={b.stat().st_size:6}")

    print("=== FORBIDDEN IN PACK ===")
    for n in FORBIDDEN:
        p = PACK / n
        if p.exists():
            print(f"BAD   {n} still in pack")
            drift = True
        else:
            print(f"OK    {n} absent")

    # Extra files in pack?
    extras = sorted({p.name for p in PACK.iterdir() if p.is_file()} - set(EXPECTED))
    if extras:
        print("EXTRA IN PACK:", extras)
        drift = True
    else:
        print("OK    no extra pack files")

    # Root README
    a, b = ROOT / "README.md", ROOT / "submission" / "CollegeName_TeamName" / "README.md"
    if a.exists() and b.exists():
        same = a.read_bytes() == b.read_bytes()
        if not same:
            drift = True
        print(("OK" if same else "DRIFT") + "  root README.md")

    # Zip vs folder
    zpath = ROOT / "submission" / "CollegeName_TeamName.zip"
    print(f"ZIP exists={zpath.exists()} size={zpath.stat().st_size if zpath.exists() else 0}")

    print("RESULT:", "DRIFT" if drift else "SYNCED")
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
