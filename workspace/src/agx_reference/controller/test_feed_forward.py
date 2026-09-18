"""Offline checks of FeedForward: python agx_reference/controller/test_feed_forward.py."""

from pathlib import Path

import numpy as np
import pinocchio as pin

from controller.feed_forward import FeedForward, fit_friction

URDF_PATH = Path(__file__).resolve().parents[1] / "piper/piper/urdf/piper_description.urdf"
DOFS = 6


def _feed_forward(f_viscous: np.ndarray | None = None, f_coulomb: np.ndarray | None = None):
    return FeedForward(str(URDF_PATH), DOFS, f_viscous=f_viscous, f_coulomb=f_coulomb)


def test_inertial_torque_equals_the_mass_matrix_times_acceleration() -> None:
    feed_forward = _feed_forward()
    model, data = feed_forward.pin_model.robot.model, feed_forward.pin_model.robot.data
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-1.5, 1.5, DOFS)
        qdd = rng.uniform(-3.0, 3.0, DOFS)
        mass_matrix = pin.crba(model, data, q)
        mass_matrix = np.triu(mass_matrix) + np.triu(mass_matrix, 1).T  # crba fills the upper half
        np.testing.assert_allclose(
            feed_forward.compute_torque(q, np.zeros(DOFS), qdd),
            mass_matrix @ qdd,
            atol=1e-10,
        )


def test_zero_acceleration_and_velocity_give_zero_torque() -> None:
    feed_forward = _feed_forward(f_viscous=np.full(DOFS, 0.5), f_coulomb=np.full(DOFS, 0.3))
    q = np.array([0.1, 0.8, -0.7, 0.2, -0.3, 0.4])
    np.testing.assert_allclose(
        feed_forward.compute_torque(q, np.zeros(DOFS), np.zeros(DOFS)), 0.0, atol=1e-12
    )


def test_friction_torque_is_viscous_plus_coulomb_well_above_the_smoothing_speed() -> None:
    f_viscous = np.array([0.1, 0.2, 0.3, 0.05, 0.05, 0.05])
    f_coulomb = np.array([0.4, 0.5, 0.3, 0.1, 0.1, 0.1])
    feed_forward = _feed_forward(f_viscous=f_viscous, f_coulomb=f_coulomb)
    qd = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    np.testing.assert_allclose(
        feed_forward.compute_torque(np.zeros(DOFS), qd, np.zeros(DOFS)),
        f_viscous * qd + f_coulomb * np.sign(qd),
        rtol=1e-6,
    )


def test_fit_friction_recovers_known_coefficients_from_noisy_samples() -> None:
    rng = np.random.default_rng(1)
    qd = np.concatenate([rng.uniform(0.05, 0.5, 500), -rng.uniform(0.05, 0.5, 500)])
    tau = 0.37 * qd + 0.21 * np.sign(qd) + rng.normal(0.0, 0.02, qd.size)
    f_viscous, f_coulomb = fit_friction(qd, tau)
    assert abs(f_viscous - 0.37) < 0.02
    assert abs(f_coulomb - 0.21) < 0.01


def test_fit_friction_ignores_an_even_offset_such_as_a_gravity_model_error() -> None:
    qd = np.concatenate([np.linspace(0.05, 0.5, 200), -np.linspace(0.05, 0.5, 200)])
    tau = 0.37 * qd + 0.21 * np.sign(qd) + 0.15  # the offset is the same in both directions
    f_viscous, f_coulomb = fit_friction(qd, tau)
    assert abs(f_viscous - 0.37) < 1e-9
    assert abs(f_coulomb - 0.21) < 1e-9


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
