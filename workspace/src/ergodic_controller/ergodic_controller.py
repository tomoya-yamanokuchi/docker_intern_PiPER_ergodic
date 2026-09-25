"""Ergodic control in tensor-train format (E2T2), stepped with a measured state.

A port of the control law in Ergodic_Exploration_using_TT_PiPER.ipynb (Shetty,
Silverio, Calinon, "Ergodic Exploration using Tensor Train"). Hardware-free: it
works on a state in [0, L]^d and knows nothing about poses or the arm; the caller
maps the measured pose into that cube and the returned reference back out.
"""

from collections.abc import Callable

import numpy as np
import tt
from tt.cross import rect_cross


def _fourier_basis_tt(x: np.ndarray, K: int, L: float) -> tuple[tt.vector, list[tt.vector]]:
    """Phi(x) as a rank-1 TT, and its gradient as one rank-1 TT per dimension."""
    k = np.arange(K)
    phi = [np.cos(np.pi * x_i * k / L).reshape(1, -1, 1) for x_i in x]
    dphi = [((-np.pi / L) * k * np.sin(np.pi * x_i * k / L)).reshape(1, -1, 1) for x_i in x]
    tt_phi = tt.vector.from_list(phi)
    tt_dphi = [tt.vector.from_list([*phi[:i], dphi[i], *phi[i + 1 :]]) for i in range(len(x))]
    return tt_phi, tt_dphi


def _reference_coefficients(
    reference_pdf: Callable[[np.ndarray], np.ndarray], d: int, K: int, N: int, L: float
) -> tt.vector:
    """Fourier coefficients W_hat of the reference pdf, by Gauss-Legendre quadrature in TT (eq. 7)."""
    x0, w0 = np.polynomial.legendre.leggauss(N)
    xn = 0.5 * L * (x0 + 1.0)
    wn = 0.5 * L * w0

    def weighted_pdf(indices: np.ndarray) -> np.ndarray:
        indices = indices.astype(int)
        return reference_pdf(xn[indices]) * np.prod(wn[indices], axis=1)

    # Restarting from a rounded result truncates the TT periodically, which the
    # E2T2 authors found faster than one long cross run.
    # Tolerance relative to the tensor's norm, which quadrature weights make tiny
    # in 6-D; an absolute one stops before the cross has found every mode. A
    # kickrank of 1 grows the rank too slowly to reach a multi-modal pdf's.
    p_init = tt.rand(N, d, r=1)
    while True:
        tt_p = rect_cross(weighted_pdf, x0=p_init, nswp=5, kickrank=1, eps=1e-2)
        converged = (tt_p - p_init).norm() < 1e-2 * tt_p.norm()
        p_init = 1 * tt_p.round(0.001)
        if converged:
            break

    # Normalise away the mass the quadrature misses, e.g. pdf outside the cube.
    ones = tt.vector.from_list([np.ones(N).reshape(1, -1, 1)] * d)
    tt_p = tt_p * (1 / tt.dot(ones, tt_p))

    # A core's quadrature sum against phi_k gives the coefficient core directly.
    phi_all = np.cos(np.pi * np.outer(xn, np.arange(K)) / L)  # (N, K)
    cores = [np.einsum("anb,nk->akb", core, phi_all) for core in tt.vector.to_list(tt_p)]
    return tt.vector.from_list(cores)


def _optimisation_weights(d: int, K: int) -> tt.vector:
    """Lambda_k = (1 + |k|^2)^(-(d+1)/2), by cross approximation."""

    def weights(indices: np.ndarray) -> np.ndarray:
        return (1 + np.linalg.norm(indices.reshape(-1, d), axis=1) ** 2) ** (-(1.0 + d) / 2.0)

    return rect_cross(weights, x0=tt.rand(K, d, r=1), nswp=15, kickrank=1, eps=1e-4)


def _pull_to_centre(
    x: np.ndarray, L: float, alpha: float, c: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis weight of the ergodic command, and a velocity back into [c, L - c].

    The weight falls smoothly from 1 to 0 across the boundary band, alpha sets
    how sharply.
    """
    weight = np.tanh(alpha * (x - c)) / 2 + np.tanh(alpha * (L - c - x)) / 2
    dx = -np.tanh(alpha * (x - c)) / 2 + np.tanh(alpha * (L - c - x)) / 2
    return weight, dx


class ErgodicController:
    # The TT of the accumulated statistics is rounded every this many steps; its
    # rank grows by one per step otherwise.
    FLUSH_EVERY = 5

    def __init__(
        self,
        reference_pdf: Callable[[np.ndarray], np.ndarray],
        d: int,
        K: int,
        N: int,
        u_max: float,
        L: float = 1.0,
    ):
        """reference_pdf maps a batch (M, d) of points in [0, L]^d to densities (M,).

        K is the number of Fourier modes per dimension, N the number of
        quadrature points per dimension, u_max the speed of the reference in
        state units per second. The coefficients are computed here, once.
        """
        self.d, self.K, self.L, self.u_max = d, K, L, u_max
        # Rounding tolerances from the E2T2 notebook: sufficient in practice.
        self.tt_w_hat = _reference_coefficients(reference_pdf, d, K, N, L).round(1e-1)
        self.tt_lambda = _optimisation_weights(d, K).round(1e-2)
        # Upper rank of the accumulated statistics: too low stalls convergence,
        # too high slows every step.
        self.rmax = int(d * (np.max(self.tt_w_hat.r) + 2))
        self.tt_wt = 0 * tt.rand(self.tt_w_hat.n, d, r=1)
        self.step_count = 0

    def step(self, x_measured: np.ndarray, dt: float) -> np.ndarray:
        """Accumulate the measured state (d,) and return the next reference x + u dt (d,)."""
        x = np.asarray(x_measured, dtype=float)
        self.step_count += 1
        tt_phi, tt_dphi = _fourier_basis_tt(x, self.K, self.L)
        self.tt_wt = self.tt_wt + tt_phi
        if (self.step_count + 1) % self.FLUSH_EVERY == 0:
            self.tt_wt = self.tt_wt.round(eps=1e-4, rmax=self.rmax)
        if (self.step_count + 1) % int(100 / self.u_max) == 0:
            # A coarser rounding; the notebook notes it only matters for d >= 7.
            self.tt_wt = self.tt_wt.round(eps=1e-1, rmax=self.rmax)

        tt_weighted_dw = (self.tt_wt - self.tt_w_hat * self.step_count) * self.tt_lambda
        b = np.array([tt.dot(tt_weighted_dw, tt_dphi_i) for tt_dphi_i in tt_dphi])
        u_ergodic = -(self.u_max / (np.linalg.norm(b) + 1e-10)) * b

        weight, u_centre = _pull_to_centre(x, self.L, alpha=20, c=self.L / 20)
        u_centre = self.u_max * u_centre / (np.linalg.norm(u_centre) + 1e-8)
        u = u_ergodic * weight + u_centre * (1 - weight)
        u = self.u_max * u / (np.linalg.norm(u) + 1e-8)
        return np.clip(x + dt * u, 0.0, self.L)

    def ergodic_metric(self) -> float:
        """Weighted distance between the time-averaged and reference coefficients."""
        tt_dw = (self.tt_wt - self.tt_w_hat * self.step_count) * (1 / self.step_count)
        return (self.tt_lambda * tt_dw).norm()
