"""Phantom-mass marks: where negative progress happened, as Fourier coefficients, and four ways to use them.

A firing's segment [t_peak, t_det] gives the mark's shape
g_k = sum w(tau) F_k(x(tau)) / sum w(tau), w(tau) = min(1, (tau - t_peak) / tau_ramp),
its mass M = alpha (t_det - t_peak), its phase phi_j (phi_tilde at firing) and its width
sigma_j = max(sigma_min, c_sigma (phi_peak - phi_det)). Its strength ramps in over T_ramp:
M_j(t) = M min(1, (t - t_fire) / T_ramp) exp(-gamma (t - t_fire)).

- inside:   M g ramped into the statistic's A (and M into Z); it decays with lambda like a visit.
- separate: c_hat = (A + P) / (Z + M_P), P = sum_j M_j(t) kappa_j(phi) g_j, kappa_j a Gaussian in phi
            of width sigma_j about phi_j (1 everywhere for global marks); decays with gamma only.
- zeroed:   no mark in the statistic; each target component near the segment is scaled by
            1 - r_j(t) mask_ij, r_j = M_j(t) / M_j, mask_ij = max over the segment of
            exp(-|mu_i - x|^2 / 2 sigma_i^2).
- negative: p_eff = p - w_neg sum_j M_j(t) g_j / mean(Z), the mark as a negative target term on the
            statistic's own scale.
"""

from dataclasses import dataclass, field

import numpy as np

from experiments.sequential_ergodic.basis import Basis, values
from experiments.sequential_ergodic.detector import Firing, NegativeProgressDetector
from experiments.sequential_ergodic.statistic import Statistic
from experiments.sequential_ergodic.target import PhaseConditionedTarget

VARIANTS = (None, "inside", "separate", "separate_global", "zeroed", "negative")


@dataclass(frozen=True)
class MarkParams:
    alpha: float = 1.0  # mass per second of segment
    tau_ramp: float = 0.5  # s, the shape's weight ramp from t_peak
    T_ramp: float = 1.0  # s, the strength's ramp from firing
    gamma: float = 0.01  # 1/s, the separate, zeroed and negative marks' decay
    c_sigma: float = 1.0
    sigma_min: float = 0.02  # phase
    w_neg: float = 1.0
    # Tag a mark with phi_tilde at its segment's peak, where the agent went wrong, rather than at
    # firing, after it backed out (the spec's phi_j). A deviation, tested as C5p in M4.
    tag_at_peak: bool = False


@dataclass(frozen=True)
class Mark:
    g: np.ndarray  # (N,)
    mass: float
    phi: float
    sigma: float  # phase; inf for a global mark
    t_fire: float
    points: np.ndarray  # (P, 2) the segment
    weights: np.ndarray  # (P,) w(tau), of sum 1

    def strength(self, t: float, params: MarkParams) -> float:
        age = t - self.t_fire
        return self.mass * min(1.0, age / params.T_ramp) * float(np.exp(-params.gamma * age))


def build_mark(
    basis: Basis,
    times: np.ndarray,  # (T,) s
    path: np.ndarray,  # (T, 2)
    firing: Firing,
    params: MarkParams,
    global_: bool = False,
) -> Mark:
    seg = (times >= firing.t_peak) & (times <= firing.t_det)
    w = np.minimum(1.0, (times[seg] - firing.t_peak) / params.tau_ramp)
    if w.sum() == 0.0:  # a segment of one sample at t_peak
        w = np.ones_like(w)
    w = w / w.sum()
    sigma = (
        np.inf
        if global_
        else max(params.sigma_min, params.c_sigma * (firing.phi_peak - firing.phi_det))
    )
    return Mark(
        g=w @ values(basis, path[seg]),
        mass=params.alpha * (firing.t_det - firing.t_peak),
        phi=firing.phi_peak if params.tag_at_peak else firing.phi_det,
        sigma=sigma,
        t_fire=firing.t_det,
        points=path[seg],
        weights=w,
    )


def separate_terms(
    marks: list[Mark], phi: float, t: float, params: MarkParams
) -> tuple[np.ndarray, float]:
    """(P (N,), M_P) of the separate statistic at phase phi."""
    P, M_P = 0.0, 0.0
    for m in marks:
        kappa = (
            1.0 if np.isinf(m.sigma) else float(np.exp(-((phi - m.phi) ** 2) / (2 * m.sigma**2)))
        )
        s = m.strength(t, params) * kappa
        P, M_P = P + s * m.g, M_P + s
    return np.asarray(P), M_P


def zeroing_scale(
    marks: list[Mark], means: np.ndarray, sigmas: np.ndarray, t: float, params: MarkParams
) -> np.ndarray:  # (n,)
    scale = np.ones(len(means))
    for m in marks:
        d2 = ((means[:, None, :] - m.points[None]) ** 2).sum(axis=2)  # (n, P)
        mask = np.exp(-d2 / (2 * sigmas[:, None] ** 2)).max(axis=1)
        scale *= 1.0 - (m.strength(t, params) / m.mass) * mask
    return scale


@dataclass
class MarkingTarget:
    """The phase-conditioned target with the detector in the loop, and the marks applied by variant.

    It is the target callable the law reads (x (2,) -> p (N,)); it is called before the statistic's
    update each step, so the marks it sets act on this step's c_hat. variant None runs the detector
    and records marks without applying them.
    """

    inner: PhaseConditionedTarget
    statistic: Statistic
    basis: Basis
    detector: NegativeProgressDetector
    variant: str | None
    params: MarkParams = MarkParams()
    marks: list[Mark] = field(default_factory=list)
    firings: list[Firing] = field(default_factory=list)
    _path: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert self.variant in VARIANTS, self.variant

    def __call__(self, x: np.ndarray) -> np.ndarray:
        dt = self.inner.dt
        n = len(self._path)
        t = n * dt
        if self.variant == "zeroed":
            target = self.inner.target
            target.component_scale = zeroing_scale(
                self.marks, target.means, target.sigmas, t, self.params
            )
        p = self.inner(x)
        self._path.append(np.asarray(x))
        per = round(self.detector.period / dt)
        if n % per == 0 and n >= per:
            self._detect(t, per, dt)
        return self._apply(p, t, dt)

    def _detect(self, t: float, per: int, dt: float) -> None:
        recent = np.array(self._path[-per - 1 :])
        speed = float(np.linalg.norm(np.diff(recent, axis=0), axis=1).sum()) / (per * dt)
        firing = self.detector.step(t, self.inner.trace[-1][1], speed)
        if firing is None:
            return
        self.firings.append(firing)
        if firing.t_det <= firing.t_peak:  # the peak is now: no segment, so no mass to mark
            return
        times = dt * np.arange(len(self._path))
        self.marks.append(
            build_mark(
                self.basis,
                times,
                np.array(self._path),
                firing,
                self.params,
                self.variant == "separate_global",
            )
        )

    def _apply(self, p: np.ndarray, t: float, dt: float) -> np.ndarray:
        if self.variant == "inside":
            for m in self.marks:
                if t - m.t_fire < self.params.T_ramp:
                    step = m.mass * dt / self.params.T_ramp
                    self.statistic.A = self.statistic.A + step * m.g
                    self.statistic.Z = self.statistic.Z + step
        elif self.variant in ("separate", "separate_global"):
            phi = self.inner.trace[-1][1]
            self.statistic.extra_A, self.statistic.extra_Z = separate_terms(
                self.marks, phi, t, self.params
            )
        elif self.variant == "negative" and self.marks:
            neg = sum(m.strength(t, self.params) * m.g for m in self.marks)
            return p - self.params.w_neg * neg / float(np.mean(self.statistic.Z))
        return p
