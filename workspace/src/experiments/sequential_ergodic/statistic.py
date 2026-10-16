"""The spatial statistic: how much time the agent has spent where, as Fourier coefficients.

One recursion covers all four variants. Per mode, A_k <- rho_k A_k + dt F_k(x) and
Z_k <- rho_k Z_k + dt with rho_k = exp(-lambda_k dt), so c_k = A_k / Z_k is a normalised
weighted average of F_k over the past: lambda = 0 is full memory, a constant lambda
exponential forgetting, a lambda per mode frequency-based forgetting. A hard reset is
full memory restarted whenever the target has moved by more than a threshold.
"""

from dataclasses import dataclass, field

import numpy as np

from experiments.sequential_ergodic.basis import Basis


@dataclass
class Statistic:
    lam: np.ndarray  # (N,) 1/s, forgetting rate per mode
    reset_threshold: float = np.inf  # |p - p at the last reset| that restarts the integral
    A: np.ndarray = field(init=False)
    Z: np.ndarray = field(init=False)
    p_reset: np.ndarray | None = field(init=False, default=None)
    # Mass held outside the recursion (separate marks), set by its owner every step: c_hat is
    # (A + extra_A) / (Z + extra_Z), and nothing of it decays here.
    extra_A: np.ndarray | float = 0.0
    extra_Z: float = 0.0

    def __post_init__(self) -> None:
        self.A, self.Z = np.zeros_like(self.lam), np.zeros_like(self.lam)

    def update(self, f: np.ndarray, dt: float, p: np.ndarray) -> None:
        """Add one step at the state whose basis values are f (N,), under target p (N,)."""
        if self.p_reset is None or np.linalg.norm(p - self.p_reset) > self.reset_threshold:
            self.A, self.Z, self.p_reset = np.zeros_like(self.A), np.zeros_like(self.Z), p.copy()
        rho = np.exp(-self.lam * dt)
        self.A = rho * self.A + dt * f
        self.Z = rho * self.Z + dt

    def coefficients(self) -> np.ndarray:  # (N,)
        return (self.A + self.extra_A) / (self.Z + self.extra_Z)


def full_memory(basis: Basis) -> Statistic:
    return Statistic(np.zeros(len(basis.k)))


def hard_reset(basis: Basis, threshold: float) -> Statistic:
    return Statistic(np.zeros(len(basis.k)), reset_threshold=threshold)


def exponential(basis: Basis, lam: float) -> Statistic:
    return Statistic(np.full(len(basis.k), lam))


def frequency_forgetting(basis: Basis, lam_lo: float, lam_hi: float, kappa: float) -> Statistic:
    """Coarse modes forget at lam_hi, fine ones at lam_lo; kappa in mode-index units."""
    k2 = (basis.k**2).sum(axis=1)
    return Statistic(lam_lo + (lam_hi - lam_lo) * np.exp(-k2 / kappa**2))
