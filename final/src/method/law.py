"""The sequential ergodic law: E2T2 with a sliding target, a forgetting statistic and a retry.

The 2-D notebook's law, written for any dimension and run online. Each step takes the
tool's measured state y in the pose metric and returns the next commanded state, one
MAX_SPEED * dt away. The phase comes from that same y: the method is sensorless.

Three things differ from the notebook, for speed and accuracy in six dimensions:
- The gradient is contracted against Lambda times each basis gradient, a TT of
  Lambda's own rank, rather than forming Lambda * (W - W_hat). The target part then
  needs no TT at all: it is a weighted sum over the datapoints' rank-1 coefficients.
- Lambda is not rounded after its cross approximation (sobolev_weights says why).
- The statistic is rounded with the E2T2 notebook's 6-D rule (rmax = d (r + 2) at
  eps 1e-4, r the static target's rank), not the 2-D notebook's exact one.
"""

from dataclasses import dataclass

import numpy as np
import tt
from scipy.special import logsumexp
from tt.cross import rect_cross

from method.pose import MAX_SPEED
from method.task import N_M, K, Task, coefficients

METHODS = {  # name -> (sliding, forgetting, retry)
    "e2t2": (False, False, False),
    "sliding": (True, True, False),
    "retry": (True, True, True),
    "recall": (True, True, False),  # the weights are method.recall's, the rest is sliding's
}
FLUSH = 5  # the statistic is rounded every this many steps


def basis(y: np.ndarray, lo: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The cosines at y (d,) and their derivatives, (d, K) each."""
    arg = np.pi * np.arange(K) * ((y - lo) / L)[:, None]
    return np.cos(arg), -(np.pi / L)[:, None] * np.arange(K) * np.sin(arg)


def sobolev_weights(d: int) -> tt.vector:
    """E2T2's Lambda_k = (1 + |k|^2)^(-(d+1)/2) as a TT, by cross approximation.

    Not rounded afterwards, as the E2T2 code rounds it at 1e-2: relative to its norm,
    which Lambda_0 = 1 dominates, that discards most of the 6-D weights for k != 0
    (rank 2, off by half a typical entry), where the cross alone is rank 5 and within 1 %.
    """
    return rect_cross(
        lambda idx: (1.0 + np.linalg.norm(idx.reshape(-1, d), axis=1) ** 2) ** (-(d + 1) / 2),
        x0=tt.vector.from_list([np.ones((1, K, 1))] * d),
        nswp=15,
        kickrank=1,
        eps=1e-4,
    )


def dot_rows(cores: list[np.ndarray], rows: list[np.ndarray]) -> np.ndarray:
    """(M,): the inner product of a TT with each rank-1 tensor rows_1[m] o ... o rows_d[m]."""
    v = np.ones((len(rows[0]), 1))
    for core, row in zip(cores, rows, strict=True):
        v = np.einsum("ma,akb,mk->mb", v, core, row)
    return v[:, 0]


def tt_weighted_sum(rows: list[np.ndarray], w: np.ndarray) -> tt.vector:
    """The TT of sum_m w_m rows_1[m] o ... o rows_d[m]; rows (M, K) per axis."""
    d, eye = len(rows), np.eye(len(w))
    cores = [(w[:, None] * rows[0]).T[None]]
    cores += [np.einsum("mk,mn->mkn", rows[i], eye) for i in range(1, d - 1)]
    cores += [rows[-1][:, :, None]]
    return tt.vector.from_list(cores).round(1e-10)


def sliding_weights(phi, phi_p, delta, rho=0.0):
    """(N_P,) of sum 1: the datapoints from the phase to 2 delta ahead; rho is the back edge's
    retreat. Nothing behind the phase: what the statistic forgets there would draw the law back."""
    w = 1.0 / (1.0 + np.exp(-(phi + 2 * delta - phi_p) / delta))
    w = w / (1.0 + np.exp(-(phi_p - phi + rho) / delta))
    return w / w.sum()


@dataclass
class Statistic:
    lam: float  # forgetting factor per step; 1 is E2T2's time average
    dt: float  # s per step
    eps: float
    rmax: int
    ttA: tt.vector = None
    Z: float = 0.0
    n: int = 0

    def add(self, phi_rows: np.ndarray) -> None:  # (d, K), the cosines at the state
        tt_phi = tt.vector.from_list([r.reshape(1, -1, 1) for r in phi_rows])
        self.ttA = tt_phi * self.dt if self.ttA is None else self.ttA * self.lam + tt_phi * self.dt
        self.Z, self.n = self.lam * self.Z + self.dt, self.n + 1
        if self.n % FLUSH == 0:
            self.ttA = self.ttA.round(self.eps, self.rmax)

    def W(self) -> tt.vector:
        return self.ttA * (1 / self.Z)


class PhaseFilter:
    """The posterior mean phase, from a Bayesian filter along the master on the state alone."""

    def __init__(self, master: np.ndarray, s: np.ndarray, h: float):
        self.master, self.h, self.phi_master = master, h, s / s[-1]
        self.arc2 = (s[:, None] - s[None]) ** 2
        self.min_sigma = s[-1] / N_M
        self.log_belief, self.y_prev = None, None

    def update(self, y: np.ndarray) -> float:
        log_lik = -((self.master - y) ** 2).sum(axis=1) / (2 * self.h**2)
        if self.log_belief is None:
            log_belief = log_lik
        else:
            sigma = max(np.linalg.norm(y - self.y_prev), self.min_sigma)
            log_belief = (
                logsumexp(self.log_belief[None] - self.arc2 / (2 * sigma**2), axis=1) + log_lik
            )
        self.log_belief = log_belief - logsumexp(log_belief)
        self.y_prev = y
        return float(np.exp(self.log_belief) @ self.phi_master)


@dataclass
class StepInfo:
    phi: float  # the phase
    stall: float  # s since the last advance
    rho: float  # the back edge's retreat, in phase
    w: np.ndarray  # (N_P,) the target's weights
    lead: float  # m, how far the next state is from the measured one


class Law:
    def __init__(self, task: Task, method: str, dt: float):
        self.task, self.dt = task, dt
        self.sliding, forgetting, self.retry = METHODS[method]
        d = task.datapoints.shape[1]
        self.lam_cores = tt.vector.to_list(sobolev_weights(d))
        self.C = coefficients(task.datapoints, task.h, task.lo, task.L)
        self.delta = task.sigma_f
        # The forgetting horizon: a half-life of one pass along the master at full speed.
        self.T_f = task.S_m / MAX_SPEED / np.log(2)
        self.lam = np.exp(-dt / self.T_f) if forgetting else 1.0
        self.uniform = np.full(len(task.datapoints), 1 / len(task.datapoints))
        self.rmax = d * (int(max(tt_weighted_sum(self.C, self.uniform).round(1e-1).r)) + 2)
        self.stat = None

    def new_trial(self) -> None:
        """A fresh phase and stall; a fresh statistic unless it is E2T2's, which is handed on."""
        if self.stat is None or self.lam < 1.0:
            self.stat = Statistic(self.lam, self.dt, 1e-4, self.rmax)
        self.filter = PhaseFilter(self.task.master, self.task.s, self.task.h)
        self.lowest, self.stall = np.inf, 0
        self.last = None  # the last step: measured state, and the step commanded from it

    def lead(self, y: np.ndarray, max_lead: float) -> float:
        """This step's length: one step at MAX_SPEED, plus what the arm left of the last.

        The part of the last step the arm did not cover along its direction is carried
        over, so a lead grows while the arm stands and its spring with it, and falls back
        to one step once the arm catches up. max_lead is the caller's bound, the law has
        no notion of force.
        """
        nominal = MAX_SPEED * self.dt
        if self.last is None:
            return nominal
        y_last, step = self.last
        length = np.linalg.norm(step)
        covered = (y - y_last) @ step / max(length, 1e-12)
        return float(np.clip(nominal + length - covered, nominal, max(nominal, max_lead)))

    def weights(self, phi: float) -> np.ndarray:  # (N_P,) of sum 1
        return (
            sliding_weights(phi, self.task.phi_p, self.delta, self.rho)
            if self.sliding
            else self.uniform
        )

    def gradient(self, phi_rows: np.ndarray, dphi_rows: np.ndarray, w: np.ndarray) -> np.ndarray:
        """(d,): the gradient of the ergodic metric's Lambda-weighted error at the state."""
        W, b = self.stat.W(), np.empty(len(phi_rows))
        for i in range(len(phi_rows)):
            rows = [dphi_rows[j] if j == i else phi_rows[j] for j in range(len(phi_rows))]
            cores = [c * r[None, :, None] for c, r in zip(self.lam_cores, rows, strict=True)]
            b[i] = tt.dot(W, tt.vector.from_list(cores)) - w @ dot_rows(cores, self.C)
        return b

    def step(self, y: np.ndarray, max_lead: float = np.inf) -> tuple[np.ndarray, StepInfo]:
        """The next commanded state from the measured one, (d,) m each, at most max_lead away.

        The phase reads y as measured; the law steps from y clamped into its cube, the only
        place its basis describes, so a start outside the cube is walked into it. The clamp
        stays one step inside each face: on a face every cosine's slope across it is zero,
        so a state held there would get no push back in.
        """
        phi = self.filter.update(y)
        self.lowest = min(self.lowest, phi)
        if phi >= self.lowest + self.delta:  # an advance
            self.lowest, self.stall = phi, 0
        else:
            self.stall += 1
        self.rho = rho = self.delta * self.stall * self.dt / self.T_f if self.retry else 0.0
        hi = self.task.lo + self.task.L
        inset = MAX_SPEED * self.dt
        y_measured, y = y, np.clip(y, self.task.lo + inset, hi - inset)
        phi_rows, dphi_rows = basis(y, self.task.lo, self.task.L)
        self.stat.add(phi_rows)
        w = self.weights(phi)
        b = self.gradient(phi_rows, dphi_rows, w)
        lead = self.lead(y_measured, max_lead)
        y_next = np.clip(y - lead * b / np.linalg.norm(b), self.task.lo, hi)
        # As run_trials walks it: from the measured state, at most the lead long.
        step = y_next - y_measured
        self.last = y_measured, step * min(1.0, lead / max(np.linalg.norm(step), 1e-12))
        return y_next, StepInfo(phi, self.stall * self.dt, rho, w, lead)
