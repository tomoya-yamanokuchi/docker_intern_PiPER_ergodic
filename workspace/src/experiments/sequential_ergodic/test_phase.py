import numpy as np

from experiments.sequential_ergodic.demos import resample
from experiments.sequential_ergodic.phase import PhaseEstimator

S = np.linspace(0.0, 1.0, 101)
MU = resample(np.array([[0.1, 0.1], [0.1, 0.9], [0.9, 0.9]]), 101)  # length 1.6
SIGMA = np.full(101, 0.03)


def test_phase_tracks_an_agent_moving_along_the_master():
    est = PhaseEstimator(S, MU, SIGMA)
    est.reset(MU[0])
    dt, speed = 0.01, 0.2 / 1.6  # phase per second at u_max = 0.2
    errors = []
    for n in range(1, 700):
        true = min(1.0, speed * n * dt)
        phi_hat, phi_tilde, _ = est.update(resample(MU, 1601)[round(true * 1600)], dt)
        if n > 100 and true < 0.95:
            errors.append(abs(phi_hat - true))
            assert phi_tilde == phi_hat  # on the master: no tube penalty
    assert max(errors) < 0.02


def test_the_projection_cannot_jump_beyond_its_window():
    est = PhaseEstimator(S, MU, SIGMA)
    est.reset(MU[0])
    est.update(MU[-1], 0.01)  # the master's end, far outside the window
    assert est.phi <= est.alpha * est.window + 1e-12


def test_leaving_the_tube_costs_kappa_d_per_tube_width():
    est = PhaseEstimator(S, MU, SIGMA)
    x = MU[30] + np.array([0.15, 0.0])  # 5 tube widths right of the vertical leg
    est.reset(x)
    phi_hat, phi_tilde, d_perp = est.update(x, 0.01)
    expected_d = np.min(np.hypot(*(MU - x).T)) / 0.03
    assert np.isclose(d_perp, expected_d)
    assert np.isclose(phi_tilde, phi_hat - est.kappa_d * (expected_d - est.d_tube))
