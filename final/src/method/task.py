"""Labelling: the master and the exploration distribution, turned into what the law runs on.

Everything here is the 2-D notebook's, written for any dimension: the master resampled
at equal arc length, each datapoint labelled with the phase of its nearest master
sample, the cube spanning the data, and the kernel width chosen by leave-one-out
likelihood. It needs the recordings only, never the arm.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

K = 10  # cosines per axis, the E2T2 paper's
N_M = 400  # master samples
# A recording's ends are the operator walking to and from the arm: samples this close to
# the first or last pose add only sensor noise to the arc length.
DWELL = 0.002  # m


def trim_dwell(y: np.ndarray) -> slice:
    """The samples of y (n, d) between leaving its first pose and reaching its last."""
    moved = np.flatnonzero(np.linalg.norm(y - y[0], axis=1) > DWELL)
    left = np.flatnonzero(np.linalg.norm(y - y[-1], axis=1) > DWELL)
    if not len(moved):
        raise ValueError("the master never leaves its first pose")
    return slice(moved[0] - 1, left[-1] + 2)


def resample(y: np.ndarray, *more: np.ndarray) -> tuple[np.ndarray, ...]:
    """y (n, d), and arrays alongside it, at N_M + 1 points of equal arc length in y; plus s."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(y, axis=0), axis=1))])
    s_grid = np.linspace(0.0, s[-1], N_M + 1)
    out = [
        np.column_stack([np.interp(s_grid, s, a[:, i]) for i in range(a.shape[1])])
        for a in (y, *more)
    ]
    return (*out, s_grid)


def phase_label(Y: np.ndarray, master: np.ndarray, phi_master: np.ndarray) -> np.ndarray:
    """(n, d) -> (n,), the phase of the nearest master sample."""
    return phi_master[np.linalg.norm(Y[:, None] - master[None], axis=2).argmin(axis=1)]


def cube(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The law's cube over the points Y (n, d): its low corner and sides, both (d,).

    Half a cosine's resolution beyond the data on either side, as the notebook's cube.
    """
    lo, side = Y.min(axis=0), np.ptp(Y, axis=0)
    margin = side / (2 * (K - 1))
    return lo - margin, side + 2 * margin


def reflected_gaussian(y, mu, h, lo, L):
    """The Gaussian of width h at mu, reflected at the faces lo, lo + L, broadcast over y and mu."""
    n = 2 * L * np.arange(-2, 3)  # images beyond |n| = 2 lie more than 3L away
    images = np.concatenate([mu[..., None] + n, 2 * lo - mu[..., None] + n], axis=-1)
    return np.exp(-((y[..., None] - images) ** 2) / (2 * h**2)).sum(-1) / (np.sqrt(2 * np.pi) * h)


def kernel_width(X: np.ndarray, lo: np.ndarray, L: np.ndarray) -> float:
    """The width h maximising the leave-one-out likelihood of the datapoints X (n, d)."""

    def loo_log_likelihood(h):
        G = np.prod(
            [
                reflected_gaussian(X[:, None, i], X[None, :, i], h, lo[i], L[i])
                for i in range(X.shape[1])
            ],
            axis=0,
        )
        np.fill_diagonal(G, 0.0)
        return np.log(G.sum(axis=1) / (len(X) - 1)).sum()

    res = minimize_scalar(
        lambda lh: -loo_log_likelihood(np.exp(lh)), bounds=np.log([1e-3, 0.3]), method="bounded"
    )
    return float(np.exp(res.x))


def coefficients(X: np.ndarray, h: float, lo: np.ndarray, L: np.ndarray) -> list[np.ndarray]:
    """Per axis (n, K): the cosine coefficients of each datapoint's reflected Gaussian."""
    k = np.arange(K)
    return [
        np.exp(-((np.pi * k * h / L[i]) ** 2) / 2) * np.cos(np.pi * k * (X[:, [i]] - lo[i]) / L[i])
        for i in range(X.shape[1])
    ]


@dataclass
class Task:
    R0: np.ndarray  # (3, 3), the master's first orientation, which the metric turns from
    master: np.ndarray  # (N_M + 1, d) m, the master at equal arc length
    q_master: np.ndarray  # (N_M + 1, 6) rad, the arm along it
    s: np.ndarray  # (N_M + 1,) m, arc length
    datapoints: np.ndarray  # (N_P, d) m, the exploration distribution
    q_datapoints: np.ndarray  # (N_P, 6) rad
    phi_p: np.ndarray  # (N_P,), the labels
    lo: np.ndarray  # (d,) m, the cube's low corner
    L: np.ndarray  # (d,) m, its sides
    h: float  # m, the kernel width

    @property
    def S_m(self) -> float:  # m, the master's length
        return float(self.s[-1])

    @property
    def phi_master(self) -> np.ndarray:
        return self.s / self.S_m

    @property
    def sigma_f(self) -> float:
        """One resolvable width in phase: a kernel, or a cosine on the coarsest axis."""
        return max(self.h, float(self.L.max()) / (K - 1)) / self.S_m

    def save(self, path: Path) -> None:
        np.savez(path, **asdict(self))

    @classmethod
    def load(cls, path: Path) -> "Task":
        with np.load(path) as f:
            return cls(**{k: f[k] for k in f.files})


def label(y_master, q_master, y_datapoints, q_datapoints, R0) -> Task:
    """The task from the master (n, d) and the datapoints (N_P, d) in the metric, with their q."""
    keep = trim_dwell(y_master)
    master, q_m, s = resample(y_master[keep], q_master[keep])
    lo, L = cube(np.vstack([master, y_datapoints]))
    h = kernel_width(y_datapoints, lo, L)
    phi_p = phase_label(y_datapoints, master, s / s[-1])
    return Task(R0, master, q_m, s, y_datapoints, q_datapoints, phi_p, lo, L, h)
