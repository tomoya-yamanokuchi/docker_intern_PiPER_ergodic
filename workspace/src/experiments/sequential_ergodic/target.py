"""The target density: Gaussians along the master path, weighted by how far progress has come.

p(x | phi) ~ sum_i w(s_i; phi) N(x; mu(s_i), sigma(s_i)^2 I) + eps_u U(x), with
w(s; phi) = [beta0 + (1 - beta0) exp(-(s - s_f)^2 / 2 sigma_f^2)] sigmoid((s_f - s) / eps)
and s_f = min(1, phi + delta). The uniform term has no coefficient beyond k = 0, so
p_k = (1 - eps_u) sum_i h_i W_ik with the components' coefficients W precomputed.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit

from experiments.sequential_ergodic.basis import Basis, gaussian_coefficients
from experiments.sequential_ergodic.phase import PhaseEstimator


@dataclass
class ExpandingTarget:
    basis: Basis
    s: np.ndarray  # (S,) master phases
    mu: np.ndarray  # (S, 2)
    sigma: np.ndarray  # (S,)
    n_components: int = 20
    beta0: float = 0.15
    delta: float = 0.08
    sigma_f: float = 0.04
    eps: float = 0.02
    eps_u: float = 0.05
    s_i: np.ndarray = field(init=False)  # (n,) the components' phases
    means: np.ndarray = field(init=False)  # (n, 2)
    sigmas: np.ndarray = field(init=False)  # (n,)
    W: np.ndarray = field(init=False)  # (n, N) their coefficients
    # (n,) multiplies each component's weight before normalising: 1, or less under a zeroing mark
    component_scale: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        idx = np.round(np.linspace(0, len(self.s) - 1, self.n_components)).astype(int)
        self.s_i, self.means, self.sigmas = self.s[idx], self.mu[idx], self.sigma[idx]
        self.component_scale = np.ones(self.n_components)
        self.W = np.array(
            [gaussian_coefficients(self.basis, self.mu[i], self.sigma[i]) for i in idx]
        )

    def weights(self, phi: float) -> np.ndarray:  # (n,), of sum 1
        s_f = min(1.0, phi + self.delta)
        bump = self.beta0 + (1 - self.beta0) * np.exp(
            -((self.s_i - s_f) ** 2) / (2 * self.sigma_f**2)
        )
        w = bump * expit((s_f - self.s_i) / self.eps) * self.component_scale
        return w / w.sum()

    def coefficients(self, phi: float) -> np.ndarray:  # (N,)
        return (1 - self.eps_u) * self.weights(phi) @ self.W

    def static_coefficients(self) -> np.ndarray:  # (N,): every component alike
        return (1 - self.eps_u) * self.W.mean(axis=0)


@dataclass
class PhaseConditionedTarget:
    """The target as the controller reads it: x (2,) -> p (N,), the phase estimated on the way.

    Static (expanding = False) still estimates and logs the phase, so every condition has a trace.
    """

    estimator: PhaseEstimator
    target: ExpandingTarget
    dt: float
    expanding: bool = True
    # () -> the highest phi_tilde allowed now, e.g. from the slider's position; None caps nothing
    cap: Callable[[], float] | None = None
    # (phi_hat, phi_tilde, d_perp) per step
    trace: list[tuple[float, float, float]] = field(default_factory=list)
    _static: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._static = self.target.static_coefficients()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        phi_hat, phi_tilde, d_perp = self.estimator.update(x, self.dt)
        if self.cap is not None:
            phi_tilde = min(phi_tilde, self.cap())
        self.trace.append((phi_hat, phi_tilde, d_perp))
        return self.target.coefficients(phi_tilde) if self.expanding else self._static
