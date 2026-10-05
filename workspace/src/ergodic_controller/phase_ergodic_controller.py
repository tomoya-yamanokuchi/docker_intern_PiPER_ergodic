"""The E2T2 law fed a phase-conditioned target and spatial statistic.

The 6-D counterpart of Ergodic_Exploration_phase_key_in_lock_2D.ipynb. A phase
phi, read off a master trajectory and never a decision variable, conditions
both halves of the law:

- the target density is the taught datapoints weighted by an asymmetric kernel
  around phi + lead, sigma_f ahead and sigma_b behind, normalised every step;
- the spatial statistic is the arm's own past states weighted by phase
  similarity, normalised to mass 1 as E2T2's time average is.

A stall freezes phi; sigma_b then widens with stall / T(phi), so earlier-phase
targets return. Hardware-free: everything is in the pipeline's [0, 1]^6 cube.

Both halves are weighted sums of rank-1 Phi(z_p), so the law is evaluated
exactly as a sum over points, b_i = sum_p a_p <Lambda, Phi(z_p) * grad_i Phi(x)>,
one matrix chain through Lambda's TT cores per point. The target changes every
step, and recomputing its coefficients by TT-cross would take seconds.
"""

from dataclasses import dataclass

import numpy as np
import tt

from direct_teaching.distribution.phase_projection import phase_weights
from ergodic_controller.ergodic_controller import _optimisation_weights, _pull_to_centre

# Phases the estimator may look back and ahead of the current phi, and the
# backward kernel's ceiling: the notebook's values.
PHASE_WINDOW = (0.05, 0.12)
SIGMA_B_MAX = 0.5
# Points weighing less than this share of the heaviest leave the sum.
WEIGHT_FLOOR = 1e-6
EVENT_STEP = 0.05  # phi between two printed progress lines


@dataclass
class PhaseTask:
    """The taught task in the controller's cube."""

    X: np.ndarray  # (M, 6) datapoint cube states
    phi: np.ndarray  # (M,) their phase labels
    X_master: np.ndarray  # (N, 6) master cube states
    phi_master: np.ndarray  # (N,) master phases, t / T
    scale: np.ndarray  # (6,) projection metric: per-axis std of X
    sigma_f: float


def _basis(Z: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:  # (..., 6) -> two (..., 6, K)
    k = np.arange(K)
    return np.cos(np.pi * Z[..., None] * k), -np.pi * k * np.sin(np.pi * Z[..., None] * k)


def lambda_gradient(
    cores: list[np.ndarray], Z: np.ndarray, a: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """b_i = sum_p a_p sum_k Lambda_k Phi_k(z_p) d_i Phi_k(x), for i over the 6 axes.

    cores are Lambda's TT cores (r, K, r'). Per point, the k-sum is a matrix chain
    through the cores with core d contracted against phi(z_pd) * phi(x_d), and
    against phi(z_pi) * phi'(x_i) on axis i; prefix and suffix products share
    the chain between the six axes.
    """
    K = cores[0].shape[1]
    phi_z, _ = _basis(Z, K)  # (P, 6, K)
    phi_x, dphi_x = _basis(x, K)  # (6, K)
    plain = [np.einsum("akb,pk->pab", c, phi_z[:, d] * phi_x[d]) for d, c in enumerate(cores)]
    prefix = [np.ones((len(Z), 1, 1))]
    for m in plain:
        prefix.append(prefix[-1] @ m)
    suffix = [np.ones((len(Z), 1, 1))]
    for m in reversed(plain):
        suffix.insert(0, m @ suffix[0])
    b = np.empty(len(cores))
    for i, c in enumerate(cores):
        derivative = np.einsum("akb,pk->pab", c, phi_z[:, i] * dphi_x[i])
        b[i] = a @ (prefix[i] @ derivative @ suffix[i + 1])[:, 0, 0]
    return b


class PhaseErgodicController:
    """step and step_count as ErgodicController, so the live loop takes either."""

    def __init__(self, task: PhaseTask, u_max: float, beta: float = 1.0, K: int = 10):
        self.task, self.u_max, self.beta = task, u_max, beta
        # Rounded as ErgodicController rounds it.
        self.tt_lambda = _optimisation_weights(task.X.shape[1], K).round(1e-2)
        self.cores = tt.vector.to_list(self.tt_lambda)
        self.phi, self.stall, self.last_progress = 0.0, 0, 0.0
        self.memory_x: list[np.ndarray] = []
        self.memory_phi: list[float] = []
        self.trace: list[tuple[float, float, float]] = []  # (phi, sigma_b, stall / T) per step
        self.step_count = 0
        # The newest target weights (M,), read by LiveView's drawing thread.
        self.target_weights: np.ndarray | None = None
        self._capped = False

    def dwell(self, phi: float) -> float:
        """T(phi): the datapoints' share of phase phi, times beta."""
        d = self.task.phi - phi
        return self.beta * float(np.exp(-(d**2) / (2 * self.task.sigma_f**2)).sum())

    def _advance_phase(self, x: np.ndarray) -> None:
        """Monotone, windowed nearest-sample projection, with sigma_f / 4 hysteresis."""
        task = self.task
        lo, hi = max(0.0, self.phi - PHASE_WINDOW[0]), min(1.0, self.phi + PHASE_WINDOW[1])
        window = (task.phi_master >= lo) & (task.phi_master <= hi)
        d2 = (((task.X_master[window] - x) / task.scale) ** 2).sum(axis=1)
        phi_hat = float(task.phi_master[window][np.argmin(d2)])
        if phi_hat > self.phi + task.sigma_f / 4:
            if int(phi_hat / EVENT_STEP) > int(self.phi / EVENT_STEP):
                print(f"phase: phi {self.phi:.3f} -> {phi_hat:.3f} at step {self.step_count}")
            self.phi = phi_hat
        if self.phi > self.last_progress + task.sigma_f:
            self.last_progress, self.stall, self._capped = self.phi, 0, False
        else:
            self.stall += 1

    def point_weights(self) -> tuple[np.ndarray, np.ndarray, float]:  # (P, 6), (P,), sigma_b
        """[datapoints; memory] and a = [-w; v]: the target negated, the statistic of mass 1."""
        task = self.task
        T = self.dwell(self.phi)
        sigma_b = min(task.sigma_f * (1 + self.stall / T), SIGMA_B_MAX)
        w = phase_weights(task.phi, self.phi, task.sigma_f, sigma_b)
        self.target_weights = w
        v = np.exp(-((np.array(self.memory_phi) - self.phi) ** 2) / (2 * task.sigma_f**2))
        Z = np.vstack([task.X, np.array(self.memory_x)])
        a = np.concatenate([-w, v / v.sum()])
        keep = np.abs(a) > WEIGHT_FLOOR * np.abs(a).max()
        return Z[keep], a[keep], sigma_b

    def step(self, x_measured: np.ndarray, dt: float) -> np.ndarray:
        """Update the phase from the measured state (6,), accumulate it, return x + u dt (6,)."""
        x = np.asarray(x_measured, dtype=float)
        self.step_count += 1
        self._advance_phase(x)
        self.memory_x.append(x)
        self.memory_phi.append(self.phi)
        Z, a, sigma_b = self.point_weights()
        self.trace.append((self.phi, sigma_b, self.stall / self.dwell(self.phi)))
        if sigma_b == SIGMA_B_MAX and not self._capped:
            self._capped = True
            print(f"phase: sigma_b at its ceiling at step {self.step_count}, phi {self.phi:.3f}")
        b = lambda_gradient(self.cores, Z, a, x)
        u_ergodic = -b / (np.linalg.norm(b) + 1e-10)
        # As ErgodicController.step: blended with a pull back into the cube near its faces.
        weight, u_centre = _pull_to_centre(x, 1.0, alpha=20, c=1.0 / 20)
        u_centre = u_centre / (np.linalg.norm(u_centre) + 1e-8)
        u = u_ergodic * weight + u_centre * (1 - weight)
        u = self.u_max * u / (np.linalg.norm(u) + 1e-8)
        return np.clip(x + dt * u, 0.0, 1.0)
