import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import load_imu_csv
from src.math_utils import (
    G,
    gravity_aligned_quat,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rotmat_from_quat,
)

gt = np.loadtxt("data/synthetic/gt_tum.txt")
imu = load_imu_csv("data/synthetic/imu_euroc.csv")
t, gyro, accel = imu["t_s"], imu["gyro"], imu["accel"]

q = gravity_aligned_quat(accel[:400].mean(0))
print("q0", q, "gt0", gt[0, 4:8])

for i in range(1, len(t)):
    dt = t[i] - t[i - 1]
    q = quat_normalize(quat_mul(q, quat_from_rotvec(gyro[i] * dt)))
    if i % 1000 == 0:
        qg = gt[i, 4:8]
        d = min(1.0, abs(float(np.dot(q, qg))))
        ang = 2 * np.degrees(np.arccos(d))
        print(f"t={t[i]-t[0]:.1f}s att_err={ang:.2f}deg")

p = np.zeros(3)
v = np.zeros(3)
q = gravity_aligned_quat(accel[:400].mean(0))
for i in range(1, len(t)):
    dt = t[i] - t[i - 1]
    q = quat_normalize(quat_mul(q, quat_from_rotvec(gyro[i] * dt)))
    R = rotmat_from_quat(q)
    aw = R @ accel[i] + np.array([0.0, 0.0, -G])
    p = p + v * dt + 0.5 * aw * dt * dt
    v = v + aw * dt
print("end p", p, "gt", gt[-1, 1:4], "err", np.linalg.norm(p - gt[-1, 1:4]))
