"""The pose metric: a tool pose as a point in R^6, with rotation turned into a length.

One metric serves the phase, the master's arc length and the law's cube, so a rotation
and a translation are weighed the same way everywhere.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

# How fast the commanded pose may travel and turn, as validated on the arm.
MAX_SPEED = 0.02  # m/s
MAX_ANGULAR_SPEED = 0.33  # rad/s
# The length at which a full-speed turn and a full-speed translation cover the same arc
# in the same time, so arc length along the master is time at full speed.
ROTATION_LENGTH = MAX_SPEED / MAX_ANGULAR_SPEED  # m/rad


def to_metric(p: np.ndarray, rot: np.ndarray, R0: np.ndarray) -> np.ndarray:
    """(n, 3) m and (n, 3, 3) -> (n, 6) m; the rotation as a full-angle rotation vector from R0."""
    return np.hstack([p, ROTATION_LENGTH * R.from_matrix(rot @ R0.T).as_rotvec()])


def to_pose(y: np.ndarray, R0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(6,) m -> the position (3,) m and orientation (3, 3) it stands for."""
    return y[:3], R.from_rotvec(y[3:] / ROTATION_LENGTH).as_matrix() @ R0
