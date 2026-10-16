"""The phase estimate: a windowed projection onto the master path, alpha-beta filtered and tube-gated.

phi_hat_raw = argmin over |s - phi_hat| <= window of |x - mu(s)| / sigma(s); d_perp is that
normalised distance minimised over all s. phi_hat is not forced to be monotone. The progress the
target reads is phi_tilde = phi_hat - kappa_d max(0, d_perp - d_tube), clipped to [0, 1].
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PhaseEstimator:
    s: np.ndarray  # (S,) master phases
    mu: np.ndarray  # (S, 2) master path
    sigma: np.ndarray  # (S,) tube width
    window: float = 0.1  # phase either side of phi_hat that the projection searches
    alpha: float = 0.1
    beta: float = 0.005
    kappa_d: float = 0.05  # phase lost per tube width beyond d_tube
    d_tube: float = 2.0  # in tube widths
    phi: float = field(init=False, default=0.0)
    rate: float = field(init=False, default=0.0)  # 1/s

    def _distances(self, x: np.ndarray) -> np.ndarray:  # (S,), in tube widths
        return np.linalg.norm(self.mu - x, axis=1) / self.sigma

    def reset(self, x: np.ndarray) -> None:
        """Start at the phase x projects to over the whole master, at rest."""
        self.phi, self.rate = float(self.s[self._distances(x).argmin()]), 0.0

    def update(self, x: np.ndarray, dt: float) -> tuple[float, float, float]:
        """(phi_hat, phi_tilde, d_perp) after measuring x (2,)."""
        dist = self._distances(x)
        near = np.abs(self.s - self.phi) <= self.window
        raw = float(self.s[near][dist[near].argmin()])
        predicted = self.phi + self.rate * dt
        residual = raw - predicted
        self.phi = float(np.clip(predicted + self.alpha * residual, 0.0, 1.0))
        self.rate += self.beta * residual / dt
        d_perp = float(dist.min())
        gated = self.phi - self.kappa_d * max(0.0, d_perp - self.d_tube)
        return self.phi, float(np.clip(gated, 0.0, 1.0)), d_perp
