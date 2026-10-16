"""The negative-progress detector on phi_tilde, stepped every `period` seconds.

CUSUM branch: with d the median of the last `median` increments of phi_tilde,
g <- max(0, g - d - K_c); it fires when g > h, phi_tilde has fallen more than delta_drop below its
running peak, and T_ref has passed since the last firing.
Stall branch: |d| / period < eps_stall for T_stall while the agent moves faster than v_min.
A firing reports t_peak, the time of phi_tilde's running maximum since the last firing (no more than
seg_max before t_det), and resets the peak there.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Firing:
    t_peak: float
    t_det: float
    phi_peak: float
    phi_det: float
    branch: str  # "cusum" or "stall"


@dataclass
class NegativeProgressDetector:
    K_c: float = 0.005
    h: float = 0.05
    delta_drop: float = 0.05
    T_ref: float = 3.0  # s
    eps_stall: float = 0.01  # phase / s
    T_stall: float = 2.0  # s
    v_min: float = 0.02  # domain units / s
    period: float = 0.1  # s
    median: int = 3
    seg_max: float = 5.0  # s
    g: float = field(init=False, default=0.0)
    stalled: int = field(init=False, default=0)  # periods, counted so no rounding decides T_stall
    t_last: float = field(init=False, default=-np.inf)
    phi_peak: float = field(init=False, default=-np.inf)
    t_peak: float = field(init=False, default=0.0)
    _phi_prev: float | None = field(init=False, default=None)
    _deltas: deque = field(init=False)

    def __post_init__(self) -> None:
        self._deltas = deque(maxlen=self.median)

    def step(self, t: float, phi: float, speed: float) -> Firing | None:
        """One period: phi_tilde now, and the agent's mean speed over the period."""
        if phi > self.phi_peak:
            self.phi_peak, self.t_peak = phi, t
        if self._phi_prev is None:
            self._phi_prev = phi
            return None
        self._deltas.append(phi - self._phi_prev)
        self._phi_prev = phi
        d = float(np.median(self._deltas))
        self.g = max(0.0, self.g - d - self.K_c)
        moving_flat = abs(d) / self.period < self.eps_stall and speed > self.v_min
        self.stalled = self.stalled + 1 if moving_flat else 0
        if t <= self.t_last + self.T_ref:
            return None
        if self.g > self.h and self.phi_peak - phi > self.delta_drop:
            return self._fire(t, phi, "cusum")
        if self.stalled >= round(self.T_stall / self.period):
            return self._fire(t, phi, "stall")
        return None

    def _fire(self, t: float, phi: float, branch: str) -> Firing:
        firing = Firing(max(self.t_peak, t - self.seg_max), t, self.phi_peak, phi, branch)
        self.g, self.stalled, self.t_last = 0.0, 0, t
        self.phi_peak, self.t_peak = phi, t
        return firing
