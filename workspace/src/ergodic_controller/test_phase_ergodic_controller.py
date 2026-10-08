"""Offline checks of PhaseErgodicController: python test_phase_ergodic_controller.py."""

from pathlib import Path

import numpy as np
import tt
from scipy.spatial.transform import Rotation

from core.agx_pinocchio import AgxPinocchio
from direct_teaching.distribution.phase_projection import tcp_poses
from direct_teaching.distribution.pose_distribution import PoseDistribution
from ergodic_controller.ergodic_controller import (
    ErgodicController,
    _fourier_basis_tt,
    _optimisation_weights,
)
from ergodic_controller.phase_ergodic_controller import (
    SIGMA_B_MAX,
    ExpandingErgodicController,
    PhaseErgodicController,
    PhaseTask,
    lambda_gradient,
)

K = 5
URDF_PATH = (
    Path(__file__).resolve().parents[1] / "agx_reference/piper/piper/urdf/piper_description.urdf"
)
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
    a, _ = phase.point_weights()
    assert np.isclose(a[a > 0].sum(), 1.0) and np.isclose(a[a < 0].sum(), -1.0)


def test_sigma_b_grows_with_the_stall_to_its_ceiling() -> None:
    task = _frozen_task(RNG.uniform(0.3, 0.7, (20, 6)))
    phase = PhaseErgodicController(task, u_max=1.0, K=K)
    T = phase.dwell(0.0)  # 20 labels at phase 0: T = 20
    for n in range(1, 400):
        phase.step(np.full(6, 0.5), 0.01)
        expected = min(task.sigma_f * (1 + n / T), SIGMA_B_MAX)
        assert np.isclose(phase.trace[-1][1], expected), (n, phase.trace[-1], expected)


def test_phase_estimate_is_monotone_and_windowed() -> None:
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
    phase.step(at(0.05), 0.01)  # behind: never back
    assert np.isclose(phase.phi, 0.10)
    phase.step(at(0.60), 0.01)  # beyond the window's 0.12 ahead: only the window's end
    assert np.isclose(phase.phi, 0.22)


def _line_task(with_master: bool) -> PhaseTask:
    """A master along x at y = 0.5, 0.1 -> 0.9, and datapoints scattered 0.2 either side of it."""
    phi_master = np.linspace(0.0, 1.0, 201)
    X_master = np.column_stack([0.1 + 0.8 * phi_master, np.full(201, 0.5)])
    rng = np.random.default_rng(5)
    phi = rng.uniform(0.0, 1.0, 120)
    X = np.column_stack([0.1 + 0.8 * phi, 0.5 + rng.uniform(-0.2, 0.2, 120)])
    return PhaseTask(
        X=X,
        phi=phi,
        P_master=X_master,
        phi_master=phi_master,
        span=np.full(2, 0.1),
        sigma_f=0.04,
        axes=2,
        X_master=X_master if with_master else None,
    )


def test_target_is_the_master_until_a_stall_hands_it_to_the_datapoints() -> None:
    task = _line_task(with_master=True)
    phase = PhaseErgodicController(task, u_max=1.0, K=K)
    M, N = len(task.X), len(task.X_master)
    phase.step(np.array([0.1, 0.5]), 0.01)
    phase.stall = 0
    a, _ = phase.point_weights()
    assert np.allclose(a[:M], 0.0) and np.isclose(a[M : M + N].sum(), -1.0)
    # Half of T(phi) into a stall: half the target on each.
    phase.stall = phase.dwell(phase.phi) / 2
    a, _ = phase.point_weights()
    assert np.isclose(a[:M].sum(), -0.5) and np.isclose(a[M : M + N].sum(), -0.5)
    # From T(phi) on the target is the datapoints' alone, as without a master.
    phase.stall = 2 * phase.dwell(phase.phi)
    a, _ = phase.point_weights()
    plain = PhaseErgodicController(_line_task(with_master=False), u_max=1.0, K=K)
    plain.phi, plain.stall, plain.memory_phi = phase.phi, phase.stall, phase.memory_phi
    a_plain, _ = plain.point_weights()
    np.testing.assert_allclose(a[:M], a_plain[:M], rtol=0, atol=1e-15)
    assert np.allclose(a[M : M + N], 0.0)


def test_an_advance_away_from_everything_taught_is_not_progress() -> None:
    """Level with the master at phi 0.1 but 40 mm beside it, datapoints reaching 20 mm."""
    beside, on_it = np.array([0.18, 0.9]), np.array([0.18, 0.5])
    for on_reference, x, stalled in (
        (0.01, beside, True),
        (np.inf, beside, False),
        (0.01, on_it, False),
    ):
        task = _line_task(with_master=True)
        task.on_reference = on_reference
        phase = PhaseErgodicController(task, u_max=1.0, K=K)
        phase.step(x, 0.01)
        assert np.isclose(phase.phi, 0.10), "the phase itself still advances"
        assert (phase.stall > 0) == stalled, (on_reference, x, phase.stall)


def _walk(speed: float, steps: int) -> tuple[PhaseErgodicController, list[int]]:
    """Along the line task's master at speed times the commanded 0.005 cube units a step."""
    task = _line_task(with_master=True)
    # sigma_f = 0.04 of phase is 0.032 cube units of this master: 6.4 commanded steps.
    task.progress_steps = 6.4
    phase = PhaseErgodicController(task, u_max=1.0, K=K)
    stalls = []
    for n in range(steps):
        phase.step(np.array([0.1 + speed * 0.005 * n, 0.5]), 0.005)
        stalls.append(phase.stall)
    return phase, stalls


def test_steady_progress_is_not_a_stall_and_standing_still_is() -> None:
    """Against the reference clock: at the commanded speed and at half of it the stall
    stays within a few steps; at rest it counts every step past the tolerance."""
    _, at_speed = _walk(1.0, 120)
    _, at_half = _walk(0.5, 240)
    assert max(at_speed) <= 2 and max(at_half) <= 4, (max(at_speed), max(at_half))
    phase, at_rest = _walk(0.0, 40)
    # No advance ever comes: the lag passes 6.4 / 4 = 1.6 on the second step.
    assert at_rest[-1] == 39 and phase.lag > 1.6


def test_rejoining_the_reference_at_a_later_phase_ends_the_stall() -> None:
    """Off the taught data the phase gains nothing; back on it, everything gained since counts."""
    task = _line_task(with_master=True)
    task.progress_steps, task.on_reference = 6.4, 0.01
    phase = PhaseErgodicController(task, u_max=1.0, K=K)
    # 40 mm beside the master, level with phases 0.1, 0.2, 0.3: the phase follows, uncredited.
    for x0 in (0.18, 0.26, 0.34):
        for _ in range(5):
            phase.step(np.array([x0, 0.9]), 0.005)
    assert np.isclose(phase.phi, 0.30) and phase.stall > 10
    phase.step(np.array([0.34, 0.5]), 0.005)  # onto the master at phase 0.3
    assert phase.stall == 0


def test_tracking_follows_a_straight_master_to_its_end() -> None:
    """Closed loop with ideal tracking, x <- step(x): on the line all the way to its end.

    Only up to arrival: there the phase stalls at 1, the target passes to the
    datapoints, and the law explores them, as it should.
    """
    phase = PhaseErgodicController(_line_task(with_master=True), u_max=1.0, K=K)
    x, path = np.array([0.1, 0.5]), []
    # 0.8 cube units at 0.005 a step is 160 steps on a straight line.
    while x[0] < 0.85 and len(path) < 200:
        x = phase.step(x, 0.005)
        path.append(x)
    path = np.array(path)
    assert len(path) < 200 and phase.phi > 0.9
    assert np.abs(path[:, 1] - 0.5).max() < 0.03


def test_expanding_law_on_a_frozen_phase_is_the_phase_law() -> None:
    """Everything reached and nothing forgotten: both weights uniform, as the phase law's there."""
    X = RNG.uniform(0.3, 0.7, (12, 6))
    phase = PhaseErgodicController(_frozen_task(X), u_max=1.0, K=K)
    expanding = ExpandingErgodicController(_frozen_task(X), 1.0, forget_window=None, K=K)
    x = np.full(6, 0.5)
    for n in range(1, 41):
        x_expanding = expanding.step(x, 0.01)
        x = phase.step(x, 0.01)
        np.testing.assert_allclose(x_expanding, x, atol=1e-12, rtol=0, err_msg=f"step {n}")


def test_expanding_target_holds_everything_reached_and_only_grows() -> None:
    """Against the notebook's r_j and front kernel, written out per datapoint."""
    task = _line_task(with_master=True)
    law = ExpandingErgodicController(task, 1.0, forget_window=None, front_share=0.25, K=K)
    M, supports = len(task.X), []
    for phi in (0.0, 0.2, 0.5, 0.97):
        law.phi = phi
        a, _ = law.point_weights()
        d = task.phi - min(1.0, phi + task.sigma_f)
        f = np.exp(-(d**2) / (2 * task.sigma_f**2))
        r = np.where(d > 0, f, 1.0)
        np.testing.assert_allclose(-a[:M], 0.75 * r / r.sum() + 0.25 * f / f.sum(), atol=1e-15)
        assert np.all(a[M:] == 0.0), "the master carries none of the target"
        supports.append(r)
    assert (np.diff(supports, axis=0) >= 0).all() and (supports[-1] == 1.0).all()


def test_expanding_statistic_is_the_discounted_time_average() -> None:
    """Against the recursion W_sum <- keep W_sum + Phi(x), n <- keep n + 1, W = W_sum / n."""
    task, window = _line_task(with_master=True), 20.0
    law = ExpandingErgodicController(task, 1.0, forget_window=window, K=K)
    states = np.random.default_rng(2).uniform(0.2, 0.8, (60, 2))
    keep, W_sum, mass = 1 - 1 / window, np.zeros((K, K)), 0.0
    for x in states:
        law.step(x, 0.005)
        k = np.arange(K)
        W_sum = keep * W_sum + np.outer(np.cos(np.pi * x[0] * k), np.cos(np.pi * x[1] * k))
        mass = keep * mass + 1.0
    a, _ = law.point_weights()
    v = a[law._fixed :]
    basis = np.cos(np.pi * states[:, :, None] * np.arange(K))  # (n, 2, K)
    W = np.einsum("n,ni,nj->ij", v, basis[:, 0], basis[:, 1])
    np.testing.assert_allclose(W, W_sum / mass, atol=1e-13, rtol=0)
    assert np.isclose(v.sum(), 1.0) and np.isclose(v[-1] / v[-21], keep**-20)


def test_expanding_law_explores_a_straight_task_to_its_end() -> None:
    """Closed loop with ideal tracking: the phase reaches the master's end. With no front
    share the path covers both sides of the line on the way, where the datapoints are;
    the front share is the forward pull, and gets there sooner."""
    steps = {}
    for share in (0.0, 0.25):
        task = _line_task(with_master=True)
        law = ExpandingErgodicController(task, 1.0, forget_window=400.0, front_share=share, K=K)
        x, path = np.array([0.1, 0.5]), []
        while law.phi < 0.99 and len(path) < 4000:
            x = law.step(x, 0.005)
            path.append(x)
        assert law.phi >= 0.99, (share, law.phi, len(path))
        steps[share] = len(path)
        if share == 0.0:
            y = np.array(path)[:, 1]
            assert y.min() < 0.4 and y.max() > 0.6, (y.min(), y.max())
    assert steps[0.25] < steps[0.0], steps


def test_phase_follows_a_pure_rotation_through_the_cube() -> None:
    """A master that only turns joint 6, so peg_tcp stays put: only the rotation term can see it.

    The clock reads the orientation back out of the cube state; the expected phase is the
    master sample the state was made from.
    """
    pin_model = AgxPinocchio(str(URDF_PATH))
    q_start = np.array([0.0, 1.0, -1.0, 0.0, 0.5, 0.0])
    q_master = np.tile(q_start, (101, 1))
    q_master[:, 5] = np.linspace(0.0, 1.0, 101)
    p, quat = tcp_poses(pin_model, q_master)
    R = Rotation.from_quat(quat, scalar_first=True).as_matrix()
    distribution = PoseDistribution(np.arange(101.0), p, R, n_components=2)
    X = np.array([distribution.pose_to_state(p_i, R_i) for p_i, R_i in zip(p, R, strict=True)])
    phi_master = np.linspace(0.0, 1.0, 101)
    task = PhaseTask(
        X=X[::5],
        phi=phi_master[::5],
        P_master=X[:, :3],
        phi_master=phi_master,
        span=distribution.upper[:3] - distribution.lower[:3],
        sigma_f=0.04,
        rotation_length=0.06,
        Q_master=quat,
        distribution=distribution,
    )
    phase = PhaseErgodicController(task, u_max=1.0, K=K)
    # Steps of 0.1, inside the window's 0.12 ahead.
    for i in (10, 20, 30, 40):
        phase.step(X[i], 0.01)
        assert np.isclose(phase.phi, phi_master[i]), (i, phase.phi)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: ok")
