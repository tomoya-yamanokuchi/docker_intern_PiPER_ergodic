"""The cosine basis of Spectral Multiscale Coverage on [0, 1]^2, and target densities in it.

F_k(x) = prod_i cos(k_i pi x_i) / h_k, with h_k making F_k of unit L2 norm on the domain,
Lambda_k = (1 + |k|^2)^(-3/2), and k = (0, 0) left out: it is 1 for every density.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import erf


@dataclass(frozen=True)
class Basis:
    k: np.ndarray  # (N, 2) mode indices, 0 <= k_i < K_i, without (0, 0)
    h: np.ndarray  # (N,) normalisers
    lam: np.ndarray  # (N,) Lambda_k


def make_basis(K: tuple[int, int] = (16, 16)) -> Basis:
    k = np.stack(np.meshgrid(np.arange(K[0]), np.arange(K[1]), indexing="ij"), axis=-1)
    k = k.reshape(-1, 2)[1:]
    h = np.sqrt(np.where(k == 0, 1.0, 0.5).prod(axis=1))
    return Basis(k, h, (1.0 + (k**2).sum(axis=1)) ** -1.5)


def values(basis: Basis, x: np.ndarray) -> np.ndarray:  # (..., 2) -> (..., N)
    return np.cos(np.pi * basis.k * np.asarray(x)[..., None, :]).prod(axis=-1) / basis.h


def gradients(basis: Basis, x: np.ndarray) -> np.ndarray:  # (2,) -> (N, 2)
    a = np.pi * basis.k * np.asarray(x)
    c, ds = np.cos(a), -np.pi * basis.k * np.sin(a)
    return np.stack([ds[:, 0] * c[:, 1], c[:, 0] * ds[:, 1]], axis=1) / basis.h[:, None]


def ergodic_metric(basis: Basis, c: np.ndarray, p: np.ndarray) -> float:
    return float(basis.lam @ (c - p) ** 2)


def _axis_moments(m: float, s: float, n: int) -> np.ndarray:  # (n,)
    """int_0^1 cos(k pi x) N(x; m, s^2) dx for k = 0..n-1, in closed form.

    e^{iwx} N(x; m, s^2) = e^{iwm - w^2 s^2 / 2} N(x; m + i w s^2, s^2), whose integral over
    [0, 1] is a difference of the normal CDF at complex arguments.
    """
    w = np.pi * np.arange(n)

    def cdf(b: float) -> np.ndarray:
        return 0.5 * (1.0 + erf((b - m - 1j * w * s**2) / (s * np.sqrt(2.0))))

    return np.real(np.exp(1j * w * m - (w * s) ** 2 / 2) * (cdf(1.0) - cdf(0.0)))


def gaussian_coefficients(basis: Basis, mean: np.ndarray, sigma: float) -> np.ndarray:  # (N,)
    """p_k of the isotropic Gaussian N(mean, sigma^2 I) truncated to the domain, of mass 1."""
    n = basis.k.max(axis=0) + 1
    mx, my = (_axis_moments(mean[i], sigma, n[i]) for i in range(2))
    return mx[basis.k[:, 0]] * my[basis.k[:, 1]] / basis.h / (mx[0] * my[0])


def mixture_coefficients(
    basis: Basis,
    means: np.ndarray,  # (C, 2)
    sigmas: np.ndarray,  # (C,)
    weights: np.ndarray,  # (C,), normalised here
) -> np.ndarray:  # (N,)
    w = np.asarray(weights, dtype=float)
    W = np.array([gaussian_coefficients(basis, m, s) for m, s in zip(means, sigmas, strict=True)])
    return (w / w.sum()) @ W
