"""Offline checks of KinematicSolver: python test_kinematic_solver.py."""

from pathlib import Path

import numpy as np
import pinocchio as pin

from kinematics.kinematic_solver import KinematicSolver

URDF = str(
    Path(__file__).resolve().parents[1] / "agx_reference/piper/piper/urdf/piper_description.urdf"
)


def _pose_error(solver: KinematicSolver, q: np.ndarray, p: np.ndarray, R: np.ndarray) -> tuple:
    p_q, R_q = solver.forward_kinematics(q)
    return np.linalg.norm(p_q - p), np.linalg.norm(pin.log3(R_q.T @ R))


def _random_configurations(solver: KinematicSolver, n: int) -> np.ndarray:
    return np.random.default_rng(0).uniform(solver.q_min, solver.q_max, (n, 6))


def test_jacobian_matches_finite_difference_of_fk() -> None:
    solver = KinematicSolver(URDF)
    h = 1e-6
    for q in _random_configurations(solver, 20):
        p, R = solver.forward_kinematics(q)
        J_numeric = np.empty((6, 6))
        for j in range(6):
            p_h, R_h = solver.forward_kinematics(q + h * np.eye(6)[j])
            J_numeric[:3, j] = (p_h - p) / h
            # Angular velocity in world axes, to match LOCAL_WORLD_ALIGNED.
            J_numeric[3:, j] = R @ pin.log3(R.T @ R_h) / h
        np.testing.assert_allclose(solver.jacobian(q), J_numeric, atol=1e-5)


def test_ik_reaches_every_fk_pose() -> None:
    solver = KinematicSolver(URDF)
    for q in _random_configurations(solver, 500):
        p, R = solver.forward_kinematics(q)
        q_ik = solver.inverse_kinematics(p, R)
        assert q_ik is not None, f"no IK solution for reachable q = {q}"
        assert np.all(q_ik >= solver.q_min) and np.all(q_ik <= solver.q_max)
        position_error, rotation_error = _pose_error(solver, q_ik, p, R)
        assert position_error <= 1e-3 and rotation_error <= 1e-3


def test_ik_seeded_at_the_answer_returns_it() -> None:
    solver = KinematicSolver(URDF)
    tol = 1e-3
    for q in _random_configurations(solver, 200):
        q_ik = solver.inverse_kinematics(*solver.forward_kinematics(q), q_init=q, tol=(tol, tol))
        assert q_ik is not None
        # A pose error within tol maps back to at most |error| / sigma_min(J) in
        # joint space, to first order; factor 2 for the nonlinearity. Any other
        # branch is a finite distance away, so near singularities it is skipped.
        sigma_min = np.linalg.svd(solver.jacobian(q), compute_uv=False)[-1]
        joint_bound = 2.0 * np.sqrt(2.0) * tol / sigma_min
        if joint_bound < 0.1:
            assert np.linalg.norm(q_ik - q) <= joint_bound, (q, q_ik, joint_bound)


def test_ik_at_and_near_wrist_singularity() -> None:
    solver = KinematicSolver(URDF)
    # j5 = 0 exactly takes the closed form's singular branch; j5 = -0.003 is
    # where the idealisation error makes every closed-form branch miss, so only
    # the polish can reach it.
    for q in (
        np.array([0.4, 1.2, -1.0, 0.3, 0.0, -0.5]),
        np.array([-0.286, 0.478, -0.022, -0.384, -0.003, 1.565]),
    ):
        p, R = solver.forward_kinematics(q)
        q_ik = solver.inverse_kinematics(p, R)
        assert q_ik is not None, f"no IK solution for reachable q = {q}"
        position_error, rotation_error = _pose_error(solver, q_ik, p, R)
        assert position_error <= 1e-3 and rotation_error <= 1e-3


def test_ik_unreachable_pose_is_none() -> None:
    solver = KinematicSolver(URDF)
    assert solver.inverse_kinematics(np.array([2.0, 0.0, 0.2]), np.eye(3)) is None


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
