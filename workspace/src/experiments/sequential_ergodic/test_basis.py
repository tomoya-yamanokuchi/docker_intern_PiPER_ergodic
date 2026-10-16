import numpy as np

from experiments.sequential_ergodic.basis import (
    gaussian_coefficients,
    gradients,
    make_basis,
    values,
)

BASIS = make_basis((16, 16))
GRID = np.linspace(0.0, 1.0, 1201)
MODES = np.r_[0:6, 15, 16, 17, 100, 136, 200, 254]  # a spread of low, mixed and high modes


def _integrate(f: np.ndarray) -> np.ndarray:  # (G, G, ...) on GRID x GRID -> (...)
    return np.trapezoid(np.trapezoid(f, GRID, axis=0), GRID, axis=0)


def _modes_on_grid() -> np.ndarray:  # (G, G, len(MODES))
    X, Y = np.meshgrid(GRID, GRID, indexing="ij")
    k = BASIS.k[MODES]
    return np.cos(np.pi * k[:, 0] * X[..., None]) * np.cos(np.pi * k[:, 1] * Y[..., None])


def test_basis_functions_have_unit_norm():
    F = _modes_on_grid() / BASIS.h[MODES]
    assert np.allclose(_integrate(F**2), 1.0, atol=1e-3)


def test_gaussian_coefficients_match_numerical_integration():
    F = _modes_on_grid() / BASIS.h[MODES]
    X, Y = np.meshgrid(GRID, GRID, indexing="ij")
    # One Gaussian well inside, one cut off by two faces of the domain.
    for mean, sigma in (((0.4, 0.6), 0.15), ((0.08, 0.95), 0.05)):
        density = np.exp(-((X - mean[0]) ** 2 + (Y - mean[1]) ** 2) / (2 * sigma**2))
        density /= _integrate(density)
        expected = _integrate(density[..., None] * F)
        assert np.allclose(
            gaussian_coefficients(BASIS, np.array(mean), sigma)[MODES], expected, atol=1e-4
        )


def test_gradients_match_finite_differences():
    x, eps = np.array([0.37, 0.81]), 1e-6
    fd = np.stack(
        [(values(BASIS, x + eps * e) - values(BASIS, x - eps * e)) / (2 * eps) for e in np.eye(2)],
        axis=1,
    )
    assert np.allclose(gradients(BASIS, x), fd, atol=1e-5)
