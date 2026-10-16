import numpy as np

from experiments.sequential_ergodic.basis import make_basis
from experiments.sequential_ergodic.statistic import (
    exponential,
    frequency_forgetting,
    full_memory,
    hard_reset,
)

BASIS = make_basis((4, 4))
N = len(BASIS.k)
DT = 0.01
F = np.random.default_rng(0).normal(size=(500, N))  # stand-ins for F_k(x(t)), one row a step
P = np.zeros(N)


def _feed(statistic, rows: np.ndarray, p: np.ndarray = P):
    for f in rows:
        statistic.update(f, DT, p)
    return statistic.coefficients()


def _discounted_mean(rows: np.ndarray, lam: float) -> np.ndarray:
    """sum_m e^{-lam (t_n - t_m)} f_m / sum_m e^{-lam (t_n - t_m)}, the newest step undiscounted."""
    age = DT * np.arange(len(rows) - 1, -1, -1)
    w = np.exp(-lam * age)
    return w @ rows / w.sum()


def test_full_memory_is_the_time_average():
    assert np.allclose(_feed(full_memory(BASIS), F), F.mean(axis=0))


def test_exponential_forgetting_is_a_discounted_average():
    assert np.allclose(_feed(exponential(BASIS, 2.0), F), _discounted_mean(F, 2.0))


def test_frequency_forgetting_discounts_each_mode_at_its_own_rate():
    statistic = frequency_forgetting(BASIS, 0.1, 3.0, 2.0)
    c = _feed(statistic, F)
    expected = [_discounted_mean(F[:, i], lam) for i, lam in enumerate(statistic.lam)]
    assert np.allclose(c, expected)


def test_frequency_forgetting_forgets_coarse_modes_fastest():
    lam = frequency_forgetting(BASIS, 0.1, 3.0, 2.0).lam
    k2 = (BASIS.k**2).sum(axis=1)
    assert np.all(np.diff(lam[np.argsort(k2)]) <= 0) and lam.max() < 3.0 and lam.min() > 0.1


def test_hard_reset_restarts_when_the_target_moves_past_the_threshold():
    statistic = hard_reset(BASIS, threshold=0.5)
    _feed(statistic, F[:200])
    c_small = _feed(statistic, F[200:300], P + 0.4 / np.sqrt(N))  # moved 0.4: kept
    assert np.allclose(c_small, F[:300].mean(axis=0))
    c_big = _feed(statistic, F[300:], P + 1.0 / np.sqrt(N))  # moved 1.0: restarted
    assert np.allclose(c_big, F[300:].mean(axis=0))
