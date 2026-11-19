#!/usr/bin/env python3
"""Convert EuRoC state_groundtruth_estimate0/data.csv to TUM trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt_csv", help="EuRoC ground truth CSV")
    ap.add_argument("out_tum", help="Output TUM file")
    args = ap.parse_args()

    # EuRoC GT: timestamp, p_RS_R_x/y/z, q_RS_w/x/y/z, v, bw, ba
    data = np.loadtxt(args.gt_csv, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    t_s = data[:, 0] * 1e-9
    p = data[:, 1:4]
    # EuRoC quaternion is w,x,y,z -> TUM wants x,y,z,w
    qw, qx, qy, qz = data[:, 4], data[:, 5], data[:, 6], data[:, 7]
    q = np.column_stack([qx, qy, qz, qw])
    out = np.column_stack([t_s, p, q])
    Path(args.out_tum).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_tum, "w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        np.savetxt(f, out, fmt="%.9f")
    print(f"wrote {args.out_tum}  poses={len(out)}")


if __name__ == "__main__":
    main()
