"""The arm's model as final/ uses it: its own URDF copy, the key frame, FK over many q."""

from pathlib import Path

import numpy as np

import core.agx_pinocchio
from core.agx_pinocchio import AgxPinocchio
from method.pose import to_metric
from method.task import Task

SRC = Path(__file__).resolve().parent
URDF_PATH = SRC / "agx_reference/piper/piper/urdf/piper_description.urdf"
TCP_FRAME = "key_tcp"  # the key's tip, 0.065 m beyond link6 on the flange axis
OUTPUT_DIR = SRC.parent / "output"

# The container's PYTHONPATH names the sandbox's agx_reference, whose URDF has no key frame
# and whose code final/ must not run on.
if not Path(core.agx_pinocchio.__file__).resolve().is_relative_to(SRC):
    raise ImportError(
        f"core.agx_pinocchio came from {core.agx_pinocchio.__file__}; run through final/src/run.sh"
    )


def tool_poses(pin: AgxPinocchio, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, 6) rad -> the tool's positions (n, 3) m and orientations (n, 3, 3)."""
    poses = [pin.forward_kinematics(q_i, TCP_FRAME) for q_i in q]
    return np.array([p for p, _ in poses]), np.array([r for _, r in poses])


def measured_state(pin: AgxPinocchio, task: Task, q: np.ndarray) -> np.ndarray:  # (6,) m
    p, rot = pin.forward_kinematics(q, TCP_FRAME)
    return to_metric(p[None], rot[None], task.R0)[0]
