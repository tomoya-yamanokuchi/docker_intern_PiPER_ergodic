"""Time-weighted GMM over demonstrated flange poses, in the ergodic controller's cube.

Hardware-free and solver-free: the caller runs FK and passes poses in. A pose
becomes the E2T2 state [p, Log_mu(q)] (eqs. 9-13), scaled per axis into [0, 1]^6,
where the GMM is fitted, so pdf is a density in the controller's coordinates.
"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp

QUATERNION_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])  # scalar first

SINGULAR_TOL = 1e-9

# Smallest span an axis is scaled over: m for position, half-angle rad for
# orientation. A still axis would otherwise scale by 1 / 0.
MIN_SPAN = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

# Added to every covariance, as scikit-learn's reg_covar: the E2T2 paper's
# minimal isotropic covariance prior. Without it a component on a dwell cluster
# collapses far below what K Fourier modes and N quadrature points resolve.
REG_COVAR = 5e-3
EM_TOL = 1e-6
EM_MAX_ITER = 200


def _quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # scalar first
    a_s, a_v, b_s, b_v = a[0], a[1:], b[0], b[1:]
    return np.concatenate([[a_s * b_s - a_v @ b_v], a_s * b_v + b_s * a_v + np.cross(a_v, b_v)])


def _time_weights(t: np.ndarray) -> np.ndarray:
    """Seconds each sample stands for: up to the next sample, the median interval for the last."""
    dt = np.diff(t)
    return np.append(dt, np.median(dt))


def _gaussian_log_pdf(X: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(covariance)
    z = solve_triangular(chol, (X - mean).T, lower=True)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (np.sum(z**2, axis=0) + log_det + X.shape[1] * np.log(2.0 * np.pi))


def _log_joint(
    X: np.ndarray, priors: np.ndarray, means: np.ndarray, covariances: np.ndarray
) -> np.ndarray:  # (M, n): log prior_k + log N(x_i; mean_k, cov_k)
    components = zip(priors, means, covariances, strict=True)
    return np.stack([np.log(p) + _gaussian_log_pdf(X, m, c) for p, m, c in components], axis=1)


def _e_step(
    X: np.ndarray, priors: np.ndarray, means: np.ndarray, covariances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Responsibilities (M, n) and the log density of each sample (M,)."""
    log_joint = _log_joint(X, priors, means, covariances)
    log_density = logsumexp(log_joint, axis=1)
    return np.exp(log_joint - log_density[:, None]), log_density


def _m_step(
    X: np.ndarray, w: np.ndarray, responsibilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Priors, means and covariances with every sum weighted by w_i r_ik; w sums to 1."""
    wr = w[:, None] * responsibilities
    mass = wr.sum(axis=0)
    means = (wr.T @ X) / mass[:, None]
    covariances = np.empty((len(mass), X.shape[1], X.shape[1]))
    for k, mean in enumerate(means):
        diff = X - mean
        covariances[k] = (wr[:, k, None] * diff).T @ diff / mass[k] + REG_COVAR * np.eye(X.shape[1])
    return mass, means, covariances


def _fit_weighted_gmm(
    X: np.ndarray, w: np.ndarray, n_components: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EM for a GMM on samples X (M, d) with weights w (M,)."""
    w = w / w.sum()
    d = X.shape[1]
    diff = X - w @ X
    covariance = (w[:, None] * diff).T @ diff + REG_COVAR * np.eye(d)
    means = X[rng.choice(len(X), n_components, replace=False, p=w)]
    covariances = np.repeat(covariance[None], n_components, axis=0)
    priors = np.full(n_components, 1.0 / n_components)
    log_likelihood = -np.inf
    for _ in range(EM_MAX_ITER):
        responsibilities, log_density = _e_step(X, priors, means, covariances)
        previous, log_likelihood = log_likelihood, w @ log_density
        if abs(log_likelihood - previous) < EM_TOL:
            break
        priors, means, covariances = _m_step(X, w, responsibilities)
    return priors, means, covariances


class PoseDistribution:
    def __init__(
        self,
        t: np.ndarray,
        p: np.ndarray,
        R: np.ndarray,
        n_components: int,
        margin: float = 0.1,
        seed: int = 0,
    ):
        """One demonstration: t (N,) s, flange position p (N, 3) m, rotation R (N, 3, 3).

        margin widens each axis beyond the data by that fraction of its span, so
        the data stays clear of the controller's pull-to-centre band.
        """
        w = _time_weights(t)
        quaternions = Rotation.from_matrix(R).as_quat(scalar_first=True)
        # Unweighted, as in the paper.
        self.mu = self.quaternion_mean(quaternions)
        S = np.hstack([p, [self.quaternion_log(q, self.mu) for q in quaternions]])

        s_min, s_max = S.min(axis=0), S.max(axis=0)
        half_width = (0.5 + margin) * np.maximum(s_max - s_min, MIN_SPAN)
        self.lower = (s_min + s_max) / 2 - half_width
        self.upper = (s_min + s_max) / 2 + half_width

        X = (S - self.lower) / (self.upper - self.lower)
        rng = np.random.default_rng(seed)
        self.priors, self.means, self.covariances = _fit_weighted_gmm(X, w, n_components, rng)

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """Density at a batch of cube states (M, 6), (M,). Not clipped to the cube."""
        log_joint = _log_joint(np.atleast_2d(X), self.priors, self.means, self.covariances)
        return np.exp(logsumexp(log_joint, axis=1))

    def marginal_pdf(self, X: np.ndarray, dims: list[int]) -> np.ndarray:
        """Density of the cube axes dims, the rest integrated out, at (M, len(dims)), (M,)."""
        means = self.means[:, dims]
        covariances = self.covariances[:, dims][:, :, dims]
        log_joint = _log_joint(np.atleast_2d(X), self.priors, means, covariances)
        return np.exp(logsumexp(log_joint, axis=1))

    def pose_to_state(self, p: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Flange position (3,) m and rotation (3, 3) to a cube state (6,)."""
        q = Rotation.from_matrix(R).as_quat(scalar_first=True)
        s = np.concatenate([p, self.quaternion_log(q, self.mu)])
        return (s - self.lower) / (self.upper - self.lower)

    def state_to_pose(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cube state (6,) to flange position (3,) m and rotation (3, 3), eq. 14."""
        s = self.lower + np.asarray(x, dtype=float) * (self.upper - self.lower)
        q = self.quaternion_exp(s[3:], self.mu)
        return s[:3], Rotation.from_quat(q, scalar_first=True).as_matrix()

    @staticmethod
    def quaternion_log(q: np.ndarray, g: np.ndarray = QUATERNION_IDENTITY) -> np.ndarray:
        """Log_g(q) = Log(conj(g) * q): unit quaternion (4,) to the tangent space at g, (3,).

        Quaternions are scalar first, [q_s, q_v]. The half-angle convention of
        the E2T2 paper: |Log(q)| is half the rotation angle, not scipy's rotvec.
        """
        g = np.asarray(g, dtype=float)
        g_conj = np.concatenate([g[:1], -g[1:]])
        q_g = _quaternion_multiply(g_conj, np.asarray(q, dtype=float))
        q_s, q_v = q_g[0], q_g[1:]
        norm_v = np.linalg.norm(q_v)
        # The paper's q_s = 1 case; testing |q_v| also covers q_s = -1, the same
        # rotation, where acos* is 0 but q_v / |q_v| is 0 / 0.
        if norm_v < SINGULAR_TOL:
            return np.zeros(3)
        # acos*: the shorter of the two antipodal representatives.
        angle = np.arccos(np.clip(q_s, -1.0, 1.0))
        if q_s < 0.0:
            angle -= np.pi
        return angle * q_v / norm_v

    @staticmethod
    def quaternion_exp(v: np.ndarray, g: np.ndarray = QUATERNION_IDENTITY) -> np.ndarray:
        """Exp_g(v) = g * Exp(v): tangent vector (3,) at g back to a unit quaternion (4,)."""
        v = np.asarray(v, dtype=float)
        norm_v = np.linalg.norm(v)
        if norm_v == 0.0:
            return np.asarray(g, dtype=float).copy()
        exp_v = np.concatenate([[np.cos(norm_v)], np.sin(norm_v) * v / norm_v])
        return _quaternion_multiply(np.asarray(g, dtype=float), exp_v)

    @staticmethod
    def quaternion_mean(
        quaternions: np.ndarray, iters: int = 100, tol: float = 1e-12
    ) -> np.ndarray:
        """Mean on the manifold of unit quaternions (M, 4): iterate v = mean Log_mu(q_i), mu <- Exp_mu(v)."""
        quaternions = np.asarray(quaternions, dtype=float)
        mu = quaternions[0]
        for _ in range(iters):
            v = np.mean([PoseDistribution.quaternion_log(q, mu) for q in quaternions], axis=0)
            mu = PoseDistribution.quaternion_exp(v, mu)
            if np.linalg.norm(v) < tol:
                break
        return mu
