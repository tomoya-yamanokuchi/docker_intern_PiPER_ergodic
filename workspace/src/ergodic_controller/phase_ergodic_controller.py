"""The E2T2 law fed a phase-conditioned target and spatial statistic.

The 6-D counterpart of Ergodic_Exploration_phase_key_in_lock_2D.ipynb. A phase
phi, read off a master trajectory and never a decision variable, conditions
both halves of the law:

- the target density is the taught datapoints weighted by an asymmetric kernel
  around phi + lead (LEAD_SCALE * sigma_f), sigma_f ahead and sigma_b behind,
  normalised every step;
- the spatial statistic is the arm's own past states weighted by phase
  similarity, normalised to mass 1 as E2T2's time average is.

A stall freezes phi; sigma_b then widens with stall / T(phi), so earlier-phase
targets return. Hardware-free: everything is in the pipeline's [0, 1]^6 cube, or
in its leading PhaseTask.axes axes -- the x-y plane for the planar run.

Both halves are weighted sums of rank-1 Phi(z_p), so the law is evaluated
exactly as a sum over points, b_i = sum_p a_p <Lambda, Phi(z_p) * grad_i Phi(x)>,
one matrix chain through Lambda's TT cores per point. The target changes every
step, and recomputing its coefficients by TT-cross would take seconds.
"""

from dataclasses import dataclass

import numpy as np
import tt
from scipy.spatial.transform import Rotation

from direct_teaching.distribution.phase_projection import phase_weights, rotation_angle
from direct_teaching.distribution.pose_distribution import PoseDistribution
from ergodic_controller.ergodic_controller import _optimisation_weights, _pull_to_centre

# Phases the estimator may look back and ahead of the current phi, and the
# backward kernel's ceiling: the notebook's values.
PHASE_WINDOW = (0.05, 0.12)
SIGMA_B_MAX = 0.5
EVENT_STEP = 0.05  # phi between two printed progress lines


@dataclass
class PhaseTask:
    """The taught task in the controller's cube."""

    X: np.ndarray  # (M, 6) datapoint cube states
    phi: np.ndarray  # (M,) their phase labels
    P_master: np.ndarray  # (N, len(span)) master cube positions
    phi_master: np.ndarray  # (N,) master phases, t / T
    span: np.ndarray  # m per cube unit on the projected axes, so the projection is in metres
    sigma_f: float
    # The leading cube axes the law runs on; 2 is the x-y plane, the rest held by the caller.
    axes: int = 6
    # l of phase_projection's pose distance, m per rad; 0 projects on position alone. The
    # rotation term needs the master's orientations and the cube to read x's back out.
    rotation_length: float = 0.0
    Q_master: np.ndarray | None = None  # (N, 4) scalar-first quaternions
    distribution: PoseDistribution | None = None


def _cosines(Z: np.ndarray, K: int) -> np.ndarray:  # (..., axes) -> (..., axes, K)
    return np.cos(np.pi * Z[..., None] * np.arange(K))


def _gradient(
    cores: list[np.ndarray],
    phi_z: np.ndarray,  # (P, axes, K), the points' cosine basis
    a: np.ndarray,  # (P,)
    x: np.ndarray,  # (axes,)
) -> np.ndarray:  # (axes,)
    """lambda_gradient from the points' basis, which the controller caches.

    Per point, the k-sum is a matrix chain through the cores with core d contracted
    against phi(z_pd) * phi(x_d), and against phi(z_pi) * phi'(x_i) on axis i;
    prefix and suffix products share the chain between the axes. The modes are
    summed against x first, so each core costs one (P, K) @ (K, r r') product.
    """
    K = cores[0].shape[1]
    k = np.arange(K)
    phi_x = np.cos(np.pi * x[:, None] * k)
    dphi_x = -np.pi * k * np.sin(np.pi * x[:, None] * k)

    def contracted(d: int, f: np.ndarray) -> np.ndarray:  # f (K,) -> (P, r, r')
        r, _, r_next = cores[d].shape
        weighted = (cores[d] * f[None, :, None]).transpose(1, 0, 2).reshape(K, r * r_next)
        return (phi_z[:, d] @ weighted).reshape(-1, r, r_next)

    plain = [contracted(d, phi_x[d]) for d in range(len(cores))]
    prefix = [np.ones((len(phi_z), 1))]
    for m in plain:
        prefix.append(np.einsum("pa,pab->pb", prefix[-1], m))
    suffix = [np.ones((len(phi_z), 1))]
    for m in reversed(plain):
        suffix.insert(0, np.einsum("pab,pb->pa", m, suffix[0]))
    b = np.empty(len(cores))
    for i in range(len(cores)):
        b[i] = a @ np.einsum("pa,pab,pb->p", prefix[i], contracted(i, dphi_x[i]), suffix[i + 1])
    return b


def lambda_gradient(
    cores: list[np.ndarray], Z: np.ndarray, a: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """b_i = sum_p a_p sum_k Lambda_k Phi_k(z_p) d_i Phi_k(x), for i over the axes.

    cores are Lambda's TT cores (r, K, r'), Z the points (P, axes).
    """
    return _gradient(cores, _cosines(Z, cores[0].shape[1]), a, x)


class PhaseErgodicController:
    """step and step_count as ErgodicController, so the live loop takes either."""

    def __init__(self, task: PhaseTask, u_max: float, beta: float = 1.0, K: int = 10):
        self.task, self.u_max, self.beta = task, u_max, beta
        # Rounded as ErgodicController rounds it.
        self.tt_lambda = _optimisation_weights(task.axes, K).round(1e-2)
        self.cores = tt.vector.to_list(self.tt_lambda)
        self.phi, self.stall, self.last_progress = 0.0, 0, 0.0
        self.memory_x: list[np.ndarray] = []
        self.memory_phi: list[float] = []
        # The cosine basis of [datapoints; memory], which does not change once a state
        # is stored: recomputing it for every past state was half of a step's cost.
        self._basis = np.empty((len(task.X) + 1024, task.axes, K))
        self._basis[: len(task.X)] = _cosines(task.X[:, : task.axes], K)
        self._cached = 0  # memory states the basis holds
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
        """Monotone, windowed nearest-sample projection by pose distance, with sigma_f / 4 hysteresis."""
        task = self.task
        lo, hi = max(0.0, self.phi - PHASE_WINDOW[0]), min(1.0, self.phi + PHASE_WINDOW[1])
        window = (task.phi_master >= lo) & (task.phi_master <= hi)
        d2 = (((task.P_master[window] - x[: len(task.span)]) * task.span) ** 2).sum(axis=1)
        if task.rotation_length:
            _, R_x = task.distribution.state_to_pose(x)
            quat = Rotation.from_matrix(R_x).as_quat(scalar_first=True)
            d2 = d2 + (task.rotation_length * rotation_angle(task.Q_master[window], quat)) ** 2
        phi_hat = float(task.phi_master[window][np.argmin(d2)])
        if phi_hat > self.phi + task.sigma_f / 4:
            if int(phi_hat / EVENT_STEP) > int(self.phi / EVENT_STEP):
                print(f"phase: phi {self.phi:.3f} -> {phi_hat:.3f} at step {self.step_count}")
            self.phi = phi_hat
        if self.phi > self.last_progress + task.sigma_f:
            self.last_progress, self.stall, self._capped = self.phi, 0, False
        else:
            self.stall += 1

    def _sync_basis(self) -> np.ndarray:  # (M + n, axes, K), rows [datapoints; memory]
        """Called once per step, after its one append: anything else is an outside edit."""
        M, n = len(self.task.X), len(self.memory_x)
        if M + n > len(self._basis):
            self._basis = np.concatenate([self._basis, np.empty_like(self._basis)])
        K = self._basis.shape[2]
        if n == self._cached + 1:
            self._basis[M + n - 1] = _cosines(self.memory_x[-1], K)
        else:
            self._basis[M : M + n] = _cosines(np.array(self.memory_x), K)
        self._cached = n
        return self._basis[: M + n]

    def point_weights(self) -> tuple[np.ndarray, float]:  # (M + n,), sigma_b
        """a = [-w; v] over [datapoints; memory]: the target negated, the statistic of mass 1."""
        task = self.task
        T = self.dwell(self.phi)
        sigma_b = min(task.sigma_f * (1 + self.stall / T), SIGMA_B_MAX)
        w = phase_weights(task.phi, self.phi, task.sigma_f, sigma_b)
        self.target_weights = w
        v = np.exp(-((np.array(self.memory_phi) - self.phi) ** 2) / (2 * task.sigma_f**2))
        return np.concatenate([-w, v / v.sum()]), sigma_b

    def step(self, x_measured: np.ndarray, dt: float) -> np.ndarray:
        """Update the phase from the measured state (axes,), accumulate it, return x + u dt (axes,)."""
        x = np.asarray(x_measured, dtype=float)
        self.step_count += 1
        self._advance_phase(x)
        self.memory_x.append(x)
        self.memory_phi.append(self.phi)
        basis = self._sync_basis()
        a, sigma_b = self.point_weights()
        self.trace.append((self.phi, sigma_b, self.stall / self.dwell(self.phi)))
        if sigma_b == SIGMA_B_MAX and not self._capped:
            self._capped = True
            print(f"phase: sigma_b at its ceiling at step {self.step_count}, phi {self.phi:.3f}")
        b = _gradient(self.cores, basis, a, x)
        u_ergodic = -b / (np.linalg.norm(b) + 1e-10)
        # As ErgodicController.step: blended with a pull back into the cube near its faces.
        weight, u_centre = _pull_to_centre(x, 1.0, alpha=20, c=1.0 / 20)
        u_centre = u_centre / (np.linalg.norm(u_centre) + 1e-8)
        u = u_ergodic * weight + u_centre * (1 - weight)
        u = self.u_max * u / (np.linalg.norm(u) + 1e-8)
        return np.clip(x + dt * u, 0.0, 1.0)
