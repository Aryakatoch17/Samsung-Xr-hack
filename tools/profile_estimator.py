import cProfile
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.estimator import run_estimator

cProfile.run(
    "run_estimator('data/degraded_imu.csv', 'outputs/estimated_trajectory.txt')",
    "outputs/prof.out",
)
p = pstats.Stats("outputs/prof.out")
p.sort_stats("cumulative").print_stats(30)
