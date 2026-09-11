"""Stamped joint-angle recording for ergodic-control target distributions.

Hardware-free: the control loop feeds samples in, the file is written once at
exit, and `load_recording` is the entry point for offline analysis.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class JointAngleRecorder:
    rec_freq_hz: float
    _t: list[float] = field(default_factory=list)
    _q: list[np.ndarray] = field(default_factory=list)
    _t0: float | None = None
    _next_slot: int = 0

    def sample(self, t_now: float, q: np.ndarray) -> None:
        """Record q if the next 1/rec_freq_hz slot is due.

        t_now: monotonic clock, s; the first call defines t = 0.
        q: (6,) rad.
        """
        if self._t0 is None:
            self._t0 = t_now
        t = t_now - self._t0
        if t < self._next_slot / self.rec_freq_hz:
            return
        # Jump past every slot already elapsed, so a loop overrun costs samples
        # instead of producing a burst of back-to-back ones to catch up.
        self._next_slot = math.floor(t * self.rec_freq_hz) + 1
        self._t.append(t)
        self._q.append(np.array(q, dtype=float))

    def __len__(self) -> int:
        return len(self._t)

    def save(self, path: Path) -> None:
        """Write t (N,) s and q (N, 6) rad to an .npz file."""
        np.savez(path, t=np.array(self._t), q=np.array(self._q).reshape(-1, 6))


def load_recording(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (t (N,) s, q (N, 6) rad) from a file written by JointAngleRecorder.save."""
    with np.load(path) as data:
        return data["t"], data["q"]
