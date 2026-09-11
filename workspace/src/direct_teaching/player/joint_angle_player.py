"""Time-indexed joint-angle targets from a recording, for replay on the arm.

Hardware-free: the control loop asks for the target at its own clock and turns
it into whatever setpoint its controller wants.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from direct_teaching.recorder.joint_angle_recorder import load_recording


@dataclass
class JointAnglePlayer:
    t: np.ndarray  # (N,) s, starts at 0
    q: np.ndarray  # (N, 6) rad

    @classmethod
    def load(
        cls, path: Path, q_start: np.ndarray, approach_speed: float = 0.3
    ) -> "JointAnglePlayer":
        """Load a recording, prefixed by a joint-space approach from q_start (6,) rad.

        The arm is almost never at the recording's first pose, and a distant
        impedance target jerks it there; the approach takes the slowest joint to
        that pose at approach_speed rad/s, and never faster than 1 s.
        """
        t_rec, q_rec = load_recording(path)
        q_start = np.asarray(q_start, dtype=float)
        t_approach = max(1.0, np.max(np.abs(q_rec[0] - q_start)) / approach_speed)
        t = np.concatenate([[0.0], t_rec - t_rec[0] + t_approach])
        return cls(t=t, q=np.vstack([q_start, q_rec]))

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def joint_angles_at(self, t: float) -> np.ndarray:  # (6,) rad
        """Linear interpolation per joint; held at the first/last sample outside [0, duration]."""
        return np.array([np.interp(t, self.t, q_j) for q_j in self.q.T])
