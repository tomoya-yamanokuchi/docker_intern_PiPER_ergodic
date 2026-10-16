import numpy as np

from experiments.sequential_ergodic.basis import gaussian_coefficients, make_basis, values
from experiments.sequential_ergodic.controller import Plant, direction, simulate
from experiments.sequential_ergodic.statistic import full_memory
from experiments.sequential_ergodic.walls import move

BASIS = make_basis((16, 16))


def test_direction_descends_the_lambda_weighted_error():
    """B is the gradient of sum_k Lambda_k e_k F_k(x), so u is its normalised descent."""
    rng = np.random.default_rng(2)
    x, e, eps = np.array([0.42, 0.31]), rng.normal(size=len(BASIS.k)), 1e-6

    def potential(y: np.ndarray) -> float:
        return BASIS.lam * e @ values(BASIS, y)

    B = np.array([(potential(x + eps * v) - potential(x - eps * v)) / (2 * eps) for v in np.eye(2)])
    u = direction(BASIS, x, e, 0.2)
    assert np.allclose(u, -0.2 * B / np.linalg.norm(B), atol=1e-6)


def test_full_memory_metric_on_a_static_gaussian_decays_at_least_like_one_over_t():
    """The rate depends on the target: the log-log slope measured here is -2.1, and -1.0 on
    M0's two-Gaussian target, so only at least 1/t is asserted."""
    plant = Plant(lambda x, d: move(x, d, np.empty((0, 2, 2))))
    p = gaussian_coefficients(BASIS, np.array([0.5, 0.5]), 0.15)
    _, metric = simulate(BASIS, full_memory(BASIS), p, np.array([0.2, 0.2]), plant, 12000)
    t = plant.dt * np.arange(1, len(metric) + 1)
    late = t >= 5.0
    slope = np.polyfit(np.log(t[late]), np.log(metric[late]), 1)[0]
    assert slope < -1.0, slope
