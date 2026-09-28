"""Offline checks of ErgodicController: python -m ergodic_controller.test_ergodic_controller.

Not as a path: the script's folder would go first on sys.path, and there
ergodic_controller.py shadows the ergodic_controller package.
"""

import numpy as np

from ergodic_controller.ergodic_controller import ErgodicController

D, K, N = 2, 8, 20
MEAN, STD = np.array([0.3, 0.6]), 0.1


def _gaussian_pdf(X: np.ndarray) -> np.ndarray:  # (M, 2) -> (M,), unnormalised
    return np.exp(-0.5 * np.sum(((X - MEAN) / STD) ** 2, axis=1))


def _phi(x: np.ndarray) -> np.ndarray:  # (2,) -> (K, K), cos(pi k x)
    k = np.arange(K)
    return np.outer(np.cos(np.pi * k * x[0]), np.cos(np.pi * k * x[1]))


def _dense_reference_coefficients() -> np.ndarray:  # (K, K)
    x0, w0 = np.polynomial.legendre.leggauss(N)
    xn, wn = 0.5 * (x0 + 1.0), 0.5 * w0
    grid = np.stack(np.meshgrid(xn, xn, indexing="ij"), axis=-1)
    density = _gaussian_pdf(grid.reshape(-1, 2)).reshape(N, N) * np.outer(wn, wn)
    density /= density.sum()
    cos = np.cos(np.pi * np.outer(xn, np.arange(K)))
    return cos.T @ density @ cos


def _dense_lambda() -> np.ndarray:  # (K, K)
    k = np.arange(K)
    return (1 + k[:, None] ** 2 + k[None, :] ** 2) ** (-(D + 1) / 2)


def test_uniform_reference_has_only_the_constant_coefficient() -> None:
    controller = ErgodicController(lambda X: np.ones(len(X)), D, K, N, u_max=1.0)
    expected = np.zeros((K, K))
    expected[0, 0] = 1.0
    np.testing.assert_allclose(controller.tt_w_hat.full(), expected, atol=1e-2)


def test_reference_coefficients_match_dense_quadrature() -> None:
    controller = ErgodicController(_gaussian_pdf, D, K, N, u_max=1.0)
    expected = _dense_reference_coefficients()
    # tt_w_hat is rounded to a relative 1e-1 in the Frobenius norm.
    error = np.linalg.norm(controller.tt_w_hat.full() - expected) / np.linalg.norm(expected)
    assert error <= 1e-1, error


def test_optimisation_weights_match_formula() -> None:
    controller = ErgodicController(_gaussian_pdf, D, K, N, u_max=1.0)
    expected = _dense_lambda()
    error = np.linalg.norm(controller.tt_lambda.full() - expected) / np.linalg.norm(expected)
    assert error <= 1e-2, error


def test_step_moves_at_u_max_and_stays_in_cube() -> None:
    u_max, dt = 1.0, 0.01
    controller = ErgodicController(_gaussian_pdf, D, K, N, u_max=u_max)
    x = np.array([0.5, 0.5])
    for _ in range(300):
        x_next = controller.step(x, dt)
        assert np.all((x_next >= 0.0) & (x_next <= 1.0))
        np.testing.assert_allclose(np.linalg.norm(x_next - x), u_max * dt, rtol=1e-6)
        x = x_next


def test_ergodic_metric_is_weighted_coefficient_distance() -> None:
    controller = ErgodicController(_gaussian_pdf, D, K, N, u_max=1.0)
    x = np.array([0.9, 0.1])
    for _ in range(300):
        x = controller.step(x, 0.01)
    # Against the controller's own accumulated statistics, not the exact sum of
    # phi: tt_wt is periodically rounded, which is a separate question.
    expected = np.linalg.norm(
        controller.tt_lambda.full()
        * (controller.tt_wt.full() / controller.step_count - controller.tt_w_hat.full())
    )
    np.testing.assert_allclose(controller.ergodic_metric(), expected, rtol=1e-6)


def test_exploration_reduces_true_ergodic_metric() -> None:
    controller = ErgodicController(_gaussian_pdf, D, K, N, u_max=1.0)
    w_hat, lam = controller.tt_w_hat.full(), _dense_lambda()
    x, phi_sum, metrics = np.array([0.9, 0.1]), np.zeros((K, K)), []
    for step in range(1, 2001):
        phi_sum += _phi(x)
        x = controller.step(x, 0.01)
        if step % 250 == 0:
            metrics.append(np.linalg.norm(lam * (phi_sum / step - w_hat)))
    # Started in a corner far from the reference mass; exploring must pull the
    # time average towards it.
    assert metrics[-1] < 0.5 * metrics[0], metrics


def test_statistics_accumulate_across_a_state_reset() -> None:
    """The property run_ergodic_trials.py rests on: a trial boundary moves the state, not W.

    Three segments with the state teleported between them, as the operator moving the
    arm back to the start pose does. Against the dense sum of phi over every state of
    every segment, computed here from the basis definition rather than from the
    controller: a controller that cleared tt_wt or step_count at a boundary would hold
    only the last segment, which the second assertion pins down.

    Kept to 90 steps so the coarse rounding at (step_count + 1) % int(100/u_max) == 0
    never fires; what remains is the eps=1e-4 rounding every fifth step.
    """
    controller = ErgodicController(_gaussian_pdf, D, K, N, u_max=1.0)
    phi_sum, phi_last = np.zeros((K, K)), np.zeros((K, K))
    for start in (np.array([0.2, 0.2]), np.array([0.8, 0.7]), np.array([0.5, 0.15])):
        phi_last = np.zeros((K, K))
        x = start
        for _ in range(30):
            phi_sum += _phi(x)
            phi_last += _phi(x)
            x = controller.step(x, 0.01)

    assert controller.step_count == 90, controller.step_count
    accumulated = controller.tt_wt.full()
    error = np.linalg.norm(accumulated - phi_sum) / np.linalg.norm(phi_sum)
    assert error < 1e-3, error
    # Teeth: resetting at each boundary would leave only the last segment's 30 states.
    reset_error = np.linalg.norm(accumulated - phi_last) / np.linalg.norm(phi_last)
    assert reset_error > 0.5, reset_error


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
