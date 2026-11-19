#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python tools/run_synthetic_benchmark.py
python tools/validate_output.py data/synthetic/imu_degraded.csv outputs/synthetic/estimated_trajectory.txt
python tools/regression_gate.py --seeds 42,7,99 || true
echo "bench done"
