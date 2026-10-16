import numpy as np

from experiments.sequential_ergodic.basis import make_basis, mixture_coefficients
from experiments.sequential_ergodic.demos import resample
from experiments.sequential_ergodic.target import ExpandingTarget

BASIS = make_basis((16, 16))
S = np.linspace(0.0, 1.0, 101)
MU = resample(np.array([[0.5, 0.15], [0.3, 0.4], [0.3, 0.6], [0.15, 0.85]]), 101)
SIGMA = np.full(101, 0.03)
TARGET = ExpandingTarget(BASIS, S, MU, SIGMA)


def test_coefficients_are_the_weighted_mixture_of_the_components():
    w = TARGET.weights(0.4)
    idx = np.round(np.linspace(0, 100, 20)).astype(int)
    expected = 0.95 * mixture_coefficients(BASIS, MU[idx], SIGMA[idx], w)
    assert np.allclose(TARGET.coefficients(0.4), expected)


def test_weights_follow_the_spec_formula():
    phi, s = 0.3, TARGET.s_i
    s_f = phi + 0.08
    w = (0.15 + 0.85 * np.exp(-((s - s_f) ** 2) / (2 * 0.04**2))) / (1 + np.exp(-(s_f - s) / 0.02))
    assert np.allclose(TARGET.weights(phi), w / w.sum())


def test_the_target_expands_with_progress():
    beyond = [TARGET.weights(phi)[TARGET.s_i > phi + 0.15].sum() for phi in (0.1, 0.5, 0.8)]
    assert max(beyond) < 1e-2
    mass_ahead = [TARGET.weights(phi) @ TARGET.s_i for phi in np.linspace(0, 1, 11)]
    assert np.all(np.diff(mass_ahead) > 0)
