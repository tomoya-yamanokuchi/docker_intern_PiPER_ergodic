"""Offline checks of FeedForward: python agx_reference/controller/test_feed_forward.py."""

from pathlib import Path

import numpy as np
import pinocchio as pin

from controller.feed_forward import GRAVITY_SCALE, FeedForward

URDF_PATH = Path(__file__).resolve().parents[1] / "piper/piper/urdf/piper_description.urdf"
DOFS = 6


F_VISCOUS = np.array([0.1, 0.2, 0.3, 0.05, 0.05, 0.05])  # (6,) N*m*s/rad
F_COULOMB = np.array([0.4, 0.5, 0.3, 0.1, 0.1, 0.1])  # (6,) N*m
F_STATIC = np.array([1.0, 1.1, 0.9, 0.2, 0.2, 0.2])  # (6,) N*m
V_STRIBECK = np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.02])  # (6,) rad/s


def _feed_forward() -> FeedForward:
    """The solver with the test coefficients, not the arm's measured ones."""
    feed_forward = FeedForward(str(URDF_PATH), DOFS)
    feed_forward.set_friction_params(F_VISCOUS, F_COULOMB, F_STATIC, V_STRIBECK)
    return feed_forward


def _gravity_correction(feed_forward: FeedForward, q: np.ndarray) -> np.ndarray:  # (6,) N*m
    """What the gravity-model term contributes at q, independent of compute_torque.

    rnea with zero velocity and zero acceleration is the gravity torque, so this
    reaches it by a different route than the class does.
    """
    model, data = feed_forward.pin_model.robot.model, feed_forward.pin_model.robot.data
    return (GRAVITY_SCALE - 1.0) * pin.rnea(model, data, q, np.zeros(DOFS), np.zeros(DOFS))


def test_inertial_torque_equals_the_mass_matrix_times_acceleration() -> None:
    feed_forward = _feed_forward()
    model, data = feed_forward.pin_model.robot.model, feed_forward.pin_model.robot.data
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-1.5, 1.5, DOFS)
        qdd = rng.uniform(-3.0, 3.0, DOFS)
        mass_matrix = pin.crba(model, data, q)
        mass_matrix = np.triu(mass_matrix) + np.triu(mass_matrix, 1).T  # crba fills the upper half
        # A still target applies no friction, which leaves the inertial term and
        # the gravity-model correction, the one part that does not vanish at rest.
        np.testing.assert_allclose(
            feed_forward.compute_torque(q, np.zeros(DOFS), np.zeros(DOFS), qdd),
            mass_matrix @ qdd + _gravity_correction(feed_forward, q),
            atol=1e-10,
        )


def test_a_resting_arm_gets_only_the_gravity_model_correction() -> None:
    """Nothing else survives zero velocity, zero acceleration and a still target."""
    feed_forward = _feed_forward()
    q = np.array([0.1, 0.8, -0.7, 0.2, -0.3, 0.4])
    np.testing.assert_allclose(
        feed_forward.compute_torque(q, np.zeros(DOFS), np.zeros(DOFS), np.zeros(DOFS)),
        _gravity_correction(feed_forward, q),
        atol=1e-12,
    )


def test_the_gravity_correction_makes_up_the_modelled_shortfall() -> None:
    """Model plus correction is GRAVITY_SCALE times the model, which is the point."""
    feed_forward = _feed_forward()
    q = np.array([0.1, 0.8, -0.7, 0.2, -0.3, 0.4])
    modelled = feed_forward.pin_model.nonlinear_effects(q, np.zeros(DOFS))
    corrected = modelled + feed_forward.compute_torque(
        q, np.zeros(DOFS), np.zeros(DOFS), np.zeros(DOFS)
    )
    np.testing.assert_allclose(corrected, GRAVITY_SCALE * modelled, atol=1e-12)


def _friction_torque(
    feed_forward: FeedForward,
    qd_cur: np.ndarray,  # (6,) rad/s
    qd_des: np.ndarray,  # (6,) rad/s
) -> np.ndarray:  # (6,) N*m
    """compute_torque at the zero pose with the gravity-model term taken back out.

    The friction checks below are about friction, and the zero pose has real
    gravity torque on joints 2, 3 and 5, so the correction would otherwise sit in
    every one of their expected values and hide what they are asserting.
    """
    zero = np.zeros(DOFS)
    return feed_forward.compute_torque(zero, qd_cur, qd_des, zero) - _gravity_correction(
        feed_forward, zero
    )


def test_a_still_arm_gets_no_friction_when_the_target_stands_still() -> None:
    """Direction comes from the desired motion, so a drifting arm is not pushed."""
    feed_forward = _feed_forward()
    qd_cur = np.array([0.3, -0.3, 0.3, -0.3, 0.3, -0.3])
    np.testing.assert_allclose(
        _friction_torque(feed_forward, qd_cur, np.zeros(DOFS)),
        0.0,
        atol=1e-12,
    )


def test_friction_torque_is_viscous_plus_coulomb_on_a_sliding_joint() -> None:
    """Well above the Stribeck velocity the static excess has decayed away."""
    feed_forward = _feed_forward()
    qd = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    np.testing.assert_allclose(
        _friction_torque(feed_forward, qd, qd),
        F_VISCOUS * qd + F_COULOMB * np.sign(qd),
        rtol=1e-6,
    )


def test_a_standing_joint_gets_the_static_level() -> None:
    """The magnitude comes from the measured speed: at rest the joint must break away."""
    feed_forward = _feed_forward()
    qd_des = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    np.testing.assert_allclose(
        _friction_torque(feed_forward, np.zeros(DOFS), qd_des),
        F_VISCOUS * qd_des + F_STATIC * np.sign(qd_des),
        rtol=1e-6,
    )


def test_static_excess_decays_as_a_gaussian_in_the_measured_speed() -> None:
    feed_forward = _feed_forward()
    qd_des = np.full(DOFS, 1.0)
    for factor in (0.5, 1.0, 2.0):
        qd_cur = factor * V_STRIBECK
        expected = F_COULOMB + (F_STATIC - F_COULOMB) * np.exp(-(factor**2))
        np.testing.assert_allclose(
            _friction_torque(feed_forward, qd_cur, qd_des),
            F_VISCOUS * qd_des + expected,
            rtol=1e-6,
        )


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
