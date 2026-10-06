"""Offline checks of PhaseErgodicController: python test_phase_ergodic_controller.py."""

import numpy as np
import tt

from ergodic_controller.ergodic_controller import (
    ErgodicController,
    _fourier_basis_tt,
    _optimisation_weights,
)
from ergodic_controller.phase_ergodic_controller import (
    SIGMA_B_MAX,
    PhaseErgodicController,
    PhaseTask,
    lambda_gradient,
)

K = 5
RNG = np.random.default_rng(0)


def _tt_sum(Z: np.ndarray, a: np.ndarray) -> tt.vector:  # sum_p a_p Phi(z_p), as a TT
    total = a[0] * _fourier_basis_tt(Z[0], K, 1.0)[0]
    for z, a_p in zip(Z[1:], a[1:], strict=True):
        total = total + a_p * _fourier_basis_tt(z, K, 1.0)[0]
    return total


def _frozen_task(X: np.ndarray) -> PhaseTask:
    """Every label at phase 0 and a master that never leaves it: the phase is frozen."""
    return PhaseTask(
        X=X,
        phi=np.zeros(len(X)),
        P_master=np.full((1, 3), 0.5),
        phi_master=np.zeros(1),
        span=np.ones(3),
        sigma_f=0.05,
    )


def test_point_sum_gradient_matches_tt_dot() -> None:
    tt_lambda = _optimisation_weights(6, K).round(1e-2)
    Z, a, x = RNG.uniform(0.1, 0.9, (7, 6)), RNG.standard_normal(7), RNG.uniform(0.1, 0.9, 6)
    _, tt_dphi = _fourier_basis_tt(x, K, 1.0)
    tt_weighted = _tt_sum(Z, a) * tt_lambda
    expected = np.array([tt.dot(tt_weighted, tt_dphi_i) for tt_dphi_i in tt_dphi])
    b = lambda_gradient(tt.vector.to_list(tt_lambda), Z, a, x)
    np.testing.assert_allclose(b, expected, rtol=1e-10, atol=1e-12)


def test_frozen_phase_steps_as_ergodic_controller() -> None:
    """With the phase frozen, target and statistic weights are uniform: E2T2 with W_hat = mean Phi(z_j)."""
    X = RNG.uniform(0.3, 0.7, (12, 6))
    phase = PhaseErgodicController(_frozen_task(X), u_max=1.0, K=K)
    reference = ErgodicController(lambda Y: np.ones(len(Y)), 6, K, 4, u_max=1.0)
    reference.tt_lambda = phase.tt_lambda
    reference.tt_w_hat = _tt_sum(X, np.full(len(X), 1 / len(X)))
    # The point sum is exact; the reference rounds its statistic every FLUSH_EVERY
    # steps, and W - ct W_hat is a small difference of two large TTs, so that
    # rounding moves its direction by far more than its relative 1e-4. Off here.
    reference.FLUSH_EVERY = 10**9
    x, dt = np.full(6, 0.5), 0.01
    # Below int(100 / u_max) steps, where the reference rounds its statistic coarsely.
    for n in range(1, 41):
        x_phase, x_reference = phase.step(x, dt), reference.step(x, dt)
        np.testing.assert_allclose(x_phase, x_reference, atol=1e-12, rtol=0, err_msg=f"step {n}")
        x = x_reference


def test_statistic_has_unit_mass_and_target_minus_one() -> None:
    phase = PhaseErgodicController(_frozen_task(RNG.uniform(0.3, 0.7, (9, 6))), u_max=1.0, K=K)
    for _ in range(6):
        phase.step(RNG.uniform(0.2, 0.8, 6), 0.01)
    _, a, _ = phase.point_weights()
    assert np.isclose(a[a > 0].sum(), 1.0) and np.isclose(a[a < 0].sum(), -1.0)


def test_sigma_b_grows_with_the_stall_to_its_ceiling() -> None:
    task = _frozen_task(RNG.uniform(0.3, 0.7, (20, 6)))
    phase = PhaseErgodicController(task, u_max=1.0, K=K)
    T = phase.dwell(0.0)  # 20 labels at phase 0: T = 20
    for n in range(1, 400):
        phase.step(np.full(6, 0.5), 0.01)
        expected = min(task.sigma_f * (1 + n / T), SIGMA_B_MAX)
        assert np.isclose(phase.trace[-1][1], expected), (n, phase.trace[-1], expected)


def test_phase_estimate_is_monotone_windowed_and_hysteretic() -> None:
    phi_master = np.linspace(0.0, 1.0, 101)
    P_master = np.full((101, 3), 0.5)
    P_master[:, 0] = 0.1 + 0.8 * phi_master
    task = PhaseTask(
        RNG.uniform(0.3, 0.7, (10, 6)),
        RNG.uniform(0, 1, 10),
        P_master,
        phi_master,
        np.full(3, 0.1),
        0.04,
    )
    phase = PhaseErgodicController(task, u_max=1.0, K=K)

    def at(phi: float) -> np.ndarray:
        return np.concatenate([P_master[round(phi * 100)], np.full(3, 0.5)])

    phase.step(at(0.10), 0.01)
    assert np.isclose(phase.phi, 0.10)
    phase.step(at(0.105), 0.01)  # within sigma_f / 4 = 0.01 of 0.10: held
    assert np.isclose(phase.phi, 0.10)
    phase.step(at(0.05), 0.01)  # behind: never back
    assert np.isclose(phase.phi, 0.10)
    phase.step(at(0.60), 0.01)  # beyond the window's 0.12 ahead: only the window's end
    assert np.isclose(phase.phi, 0.22)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: ok")
