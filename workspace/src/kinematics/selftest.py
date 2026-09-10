#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Acceptance checks for the kinematics module.

    cd workspace/src && python -m kinematics.selftest

Runs entirely on the model. It never opens a CAN socket or imports piper_sdk,
so the arm can be unpowered or unplugged -- check_no_hardware() enforces that
rather than leaving it to good intentions.
"""

import sys
import time

import numpy as np
import pinocchio as pin

from kinematics.dh import fk_dh
from kinematics.dynamics import inverse_dynamics_tau
from kinematics.fk import fk, jacobian, joint_frames, pose_error
from kinematics.ik import ik
from kinematics.ik_closed import ik_closed
from kinematics.model import Q_READY, URDF_GRIPPER_MASS, URDF_ZERO_GRIPPER_MASS, PiperKinematics
from kinematics.transform import Transform, log3, log6, matrix_to_rpy, rpy_to_matrix

SAMPLES = 5000


def check_no_hardware():
    """
    The whole package must be importable without pulling in anything that can
    reach the arm.
    """
    forbidden = sorted(m for m in ("piper_sdk", "can") if m in sys.modules)

    if forbidden:
        raise AssertionError(
            "kinematics must not import hardware modules, but found: " + ", ".join(forbidden)
        )

    print("PASS: no hardware modules imported")


def check_model(kin):
    assert kin.model.nq == 6, kin.model.nq
    assert kin.model_full.nq == 8, kin.model_full.nq

    for name in ("gripper_base", "joint7", "link7", "joint8", "link8"):
        assert kin.model.existFrame(name), f"{name} lost in buildReducedModel"

    assert kin.in_limits(Q_READY), "Q_READY is outside the joint limits"

    margin = np.minimum(Q_READY - kin.q_min, kin.q_max - Q_READY)
    assert margin.min() > 0.3, margin

    print(
        f"PASS: nq {kin.model.nq} reduced / {kin.model_full.nq} full, "
        f"gripper frames kept, Q_READY margin {margin.min():.3f} rad"
    )


def check_zero_pose(kin):
    """
    q = 0 folds the arm in against the base. This documents the convention and
    trips if the wrong URDF is ever loaded -- it says nothing about whether the
    pose is collision free or a good seed, because it is neither.
    """
    p = fk(kin, np.zeros(6)).translation

    assert np.linalg.norm(p) < 0.25, p

    print(
        f"PASS: zero pose collapsed, tcp at {np.round(p, 4)} "
        f"({np.linalg.norm(p) * 1000:.0f} mm from the base)"
    )


def check_jacobian(kin):
    """
    The analytic frame Jacobian against finite differences.
    """
    rng = np.random.default_rng(3)
    worst = 0.0

    for _ in range(20):
        q = rng.uniform(kin.q_min, kin.q_max)
        J = jacobian(kin, q, local=True)

        numeric = np.zeros((6, 6))
        base = fk(kin, q)

        for i in range(6):
            dq = np.zeros(6)
            dq[i] = 1e-7
            numeric[:, i] = log6(base.actInv(fk(kin, q + dq))) / 1e-7

        worst = max(worst, np.abs(J - numeric).max())

    assert worst < 1e-4, worst

    print(f"PASS: Jacobian matches finite differences, worst {worst:.2e}")


def check_singularities(kin):
    """
    j5 = 0 aligns the joint 4 and joint 6 axes, so the zero and mid-range
    configurations are both singular. Q_READY is deliberately not.
    """

    def sigma_min(q):
        return np.linalg.svd(jacobian(kin, q), compute_uv=False)[-1]

    mid = (kin.q_min + kin.q_max) / 2.0

    assert sigma_min(np.zeros(6)) < 1e-3
    assert sigma_min(mid) < 1e-3
    assert sigma_min(Q_READY) > 0.1

    print(
        f"PASS: sigma_min  zero {sigma_min(np.zeros(6)):.2e}  "
        f"mid-range {sigma_min(mid):.2e}  Q_READY {sigma_min(Q_READY):.3f}"
    )


def check_urdf_variants():
    """
    The two gripper-mass variants must differ in inertia and in nothing else.

    This is the check that catches a stray edit to an <origin> while changing a
    <mass>: the masses are obviously different, so only the geometry needs
    guarding, and nothing else would notice it moving.
    """
    loaded = PiperKinematics(urdf_path=URDF_GRIPPER_MASS)
    bare = PiperKinematics(urdf_path=URDF_ZERO_GRIPPER_MASS)

    assert [f.name for f in loaded.model.frames] == [f.name for f in bare.model.frames]

    # Compare the transforms themselves, not a pose_error: log6(A.actInv(B))
    # returns ~1e-16 even for identical inputs, because R.T @ R is only
    # identity to rounding. Bit equality is both stricter and the actual claim.
    for j in range(1, loaded.model.njoints):
        assert np.array_equal(
            loaded.model.jointPlacements[j].homogeneous, bare.model.jointPlacements[j].homogeneous
        ), j

    rng = np.random.default_rng(9)

    for _ in range(200):
        q = rng.uniform(bare.q_min, bare.q_max)
        assert np.array_equal(fk(loaded, q).homogeneous, fk(bare, q).homogeneous), q

    mass_loaded = pin.computeTotalMass(loaded.model)
    mass_bare = pin.computeTotalMass(bare.model)

    assert abs(mass_loaded - 3.65) < 1e-6, mass_loaded
    assert abs(mass_bare - 3.15) < 1e-6, mass_bare

    print(
        f"PASS: URDF variants kinematically identical, "
        f"mass {mass_loaded:.3f} kg loaded / {mass_bare:.3f} kg bare"
    )


def check_dynamics(kin):
    """
    The static feedforward must equal the gradient of the potential energy, and
    the vertical-axis joints must carry no gravity torque at all.
    """
    rng = np.random.default_rng(12)
    worst = 0.0
    peak = np.zeros(6)

    for _ in range(200):
        q = rng.uniform(kin.q_min, kin.q_max)
        tau = inverse_dynamics_tau(kin, q)
        peak = np.maximum(peak, np.abs(tau))

        numeric = np.zeros(6)

        for i in range(6):
            dq = np.zeros(6)
            dq[i] = 1e-6
            up = pin.computePotentialEnergy(kin.model, kin.data, pin.integrate(kin.model, q, dq))
            down = pin.computePotentialEnergy(kin.model, kin.data, pin.integrate(kin.model, q, -dq))
            numeric[i] = (up - down) / 2e-6

        worst = max(worst, np.abs(tau - numeric).max())

    assert worst < 1e-6, worst

    # Joints 1 and 6 rotate about vertical axes, so gravity cannot load them.
    assert peak[0] < 1e-3 and peak[5] < 1e-3, peak

    print(
        f"PASS: static tau matches d(PE)/dq to {worst:.2e} Nm, "
        f"peak |tau| per joint {np.round(peak, 2)}"
    )


def check_against_pinocchio(kin):
    """
    Our kinematics against the Pinocchio calls they replaced.

    This is the migration's safety net and only works while Pinocchio is still
    installed for dynamics.py. It is the one check that would catch a rewrite
    that is self-consistent but wrong -- the round trip compares our FK against
    our own IK, so it cannot.
    """
    rng = np.random.default_rng(11)
    worst = {
        "fk pos": 0.0,
        "fk rot": 0.0,
        "joint frames": 0.0,
        "J local": 0.0,
        "J world": 0.0,
        "log6": 0.0,
    }

    for _ in range(2000):
        q = rng.uniform(kin.q_min, kin.q_max)

        pin.forwardKinematics(kin.model, kin.data, q)
        pin.updateFramePlacements(kin.model, kin.data)

        ours = joint_frames(kin, q)
        reference = kin.data.oMf[kin.frame_id]

        worst["fk pos"] = max(
            worst["fk pos"], np.abs(ours[-1].translation - reference.translation).max()
        )
        worst["fk rot"] = max(worst["fk rot"], np.abs(ours[-1].rotation - reference.rotation).max())

        for joint in range(1, 7):
            worst["joint frames"] = max(
                worst["joint frames"],
                np.abs(ours[joint - 1].homogeneous - kin.data.oMi[joint].homogeneous).max(),
            )

        worst["J local"] = max(
            worst["J local"],
            np.abs(
                jacobian(kin, q, local=True)
                - pin.computeFrameJacobian(kin.model, kin.data, q, kin.frame_id, pin.LOCAL)
            ).max(),
        )
        worst["J world"] = max(
            worst["J world"],
            np.abs(
                jacobian(kin, q, local=False)
                - pin.computeFrameJacobian(
                    kin.model, kin.data, q, kin.frame_id, pin.LOCAL_WORLD_ALIGNED
                )
            ).max(),
        )

        delta = Transform(pin.exp3(rng.normal(size=3) * 0.4), rng.normal(size=3) * 0.2)
        worst["log6"] = max(
            worst["log6"],
            np.abs(log6(delta) - pin.log6(pin.SE3(delta.rotation, delta.translation)).vector).max(),
        )

    # A half turn is the normal case here, not an edge case: a tool pointing
    # down is rpy(0, pi, 0). The rotation vector is discontinuous there, so it
    # is checked explicitly rather than left to random sampling.
    for angle in (0.0, 1e-12, 1e-6, 1e-4, 1e-2, 3.0, np.pi - 1e-9):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        delta = Transform(pin.exp3(angle * axis), rng.normal(size=3) * 0.2)
        worst["log6"] = max(
            worst["log6"],
            np.abs(log6(delta) - pin.log6(pin.SE3(delta.rotation, delta.translation)).vector).max(),
        )

    # rpy is how _idealise() rebuilds every joint placement, so a convention
    # mismatch between scipy and Pinocchio would silently move the whole model.
    for joint in range(1, 7):
        rotation = kin.placement[joint].rotation
        worst["joint frames"] = max(
            worst["joint frames"], np.abs(rpy_to_matrix(*matrix_to_rpy(rotation)) - rotation).max()
        )

    for name, value in worst.items():
        assert value < 1e-9, (name, value)

    print(
        "PASS: matches Pinocchio  "
        + "  ".join(f"{name} {value:.1e}" for name, value in worst.items())
    )


def check_round_trip(kin):
    """
    Sample reachable configurations, go q -> pose -> q -> pose, and compare the
    poses. The joint angles are not expected to match: several configurations
    reach the same pose, so a different branch is a correct answer.
    """
    rng = np.random.default_rng(0)

    solved = 0
    worst_p = worst_r = 0.0
    times = []

    for _ in range(SAMPLES):
        target = fk(kin, rng.uniform(kin.q_min, kin.q_max))

        start = time.perf_counter()
        q = ik(kin, target)
        times.append(time.perf_counter() - start)

        if q is None:
            continue

        assert kin.in_limits(q), q

        p, r = pose_error(fk(kin, q), target)
        worst_p, worst_r = max(worst_p, p), max(worst_r, r)
        solved += 1

    times = np.array(times) * 1000.0
    rate = 100.0 * solved / SAMPLES

    print(
        f"PASS: round trip {solved}/{SAMPLES} = {rate:.2f}%  "
        f"worst pos {worst_p * 1e6:.3f} um, rot {np.degrees(worst_r) * 1e3:.4f} mdeg  "
        f"median {np.median(times):.2f} ms, p95 {np.percentile(times, 95):.1f} ms"
    )

    assert solved == SAMPLES, f"{SAMPLES - solved} reachable poses unsolved"

    # The closed form is solved on the idealised arm, so ~90 um is by design;
    # branch selection by proximity to q_init can pick a slightly worse one.
    # This is a regression guard, not the solver's contract, which is tol_pos.
    assert worst_p < 2.5e-4, worst_p


def check_closed_form_agrees(kin):
    """
    Every closed-form branch must already reach the target pose.

    This is what catches a sign error in the ZYZ extraction: a wrong branch is
    wrong by radians, not by the 90 um the idealised geometry costs.
    """
    rng = np.random.default_rng(11)
    worst = 0.0
    branches = 0

    for _ in range(300):
        target = fk(kin, rng.uniform(kin.q_min, kin.q_max))

        solutions = ik_closed(kin, target, limits_only=False)
        assert solutions, "closed form returned nothing for a reachable pose"

        for q in solutions:
            p, _ = pose_error(fk(kin, q), target)
            worst = max(worst, p)
            assert p < 2e-4, p

        branches += len(solutions)

    print(
        f"PASS: all {branches} closed-form branches reach their target, worst {worst * 1e6:.1f} um"
    )


def check_branch_consistency(kin):
    """
    Seeding each solve from the previous solution should keep a Cartesian path
    on one elbow and wrist branch instead of reconfiguring mid-move.
    """
    start = fk(kin, Q_READY)
    q = Q_READY.copy()
    worst_jump = 0.0

    for step in range(1, 41):
        target = Transform(start.rotation, start.translation + np.array([0.0, 0.002 * step, 0.0]))

        nxt = ik(kin, target, q_init=q)
        assert nxt is not None, f"lost the path at step {step}"

        worst_jump = max(worst_jump, np.abs(nxt - q).max())
        q = nxt

    assert worst_jump < 0.15, worst_jump

    print(f"PASS: branch consistent along a 80 mm path, worst joint step {worst_jump:.4f} rad")


def check_determinism(kin):
    """
    There is no randomness left in the solver, so this guards against any being
    reintroduced without anyone noticing.
    """
    targets = [fk(kin, q) for q in (Q_READY, np.array([0.5, 1.2, -0.9, 0.3, 0.6, -0.4]))]

    first = [ik(kin, T) for T in targets]
    again = [ik(kin, T) for T in targets]

    for a, b in zip(first, again, strict=True):
        assert np.array_equal(a, b)

    print("PASS: repeated solves return identical results")


def check_dh_tables(kin):
    """
    Both published AgileX DH tables against the URDF model.

    An independent description of the same arm, so it catches a wrong URDF
    variant or a frame convention mistake without needing the hardware.
    """
    rng = np.random.default_rng(4)
    worst = {}

    for convention in ("standard", "modified"):
        worst_p = worst_r = 0.0

        for _ in range(200):
            q = rng.uniform(kin.q_min, kin.q_max)
            T = fk_dh(q, convention)
            urdf = fk(kin, q)

            worst_p = max(worst_p, np.linalg.norm(T[:3, 3] - urdf.translation))
            worst_r = max(worst_r, np.linalg.norm(log3(T[:3, :3].T @ urdf.rotation)))

        worst[convention] = (worst_p, worst_r)

        assert worst_p < 2e-4, (convention, worst_p)
        assert np.degrees(worst_r) < 0.01, (convention, worst_r)

    print(
        "PASS: DH cross-check  "
        + "  ".join(
            f"{name} {p * 1000:.3f} mm / {np.degrees(r) * 1e3:.2f} mdeg"
            for name, (p, r) in worst.items()
        )
    )


def main():
    check_no_hardware()

    kin = PiperKinematics()

    check_urdf_variants()
    check_model(kin)
    check_zero_pose(kin)
    check_dynamics(kin)
    check_jacobian(kin)
    check_singularities(kin)
    check_dh_tables(kin)
    check_against_pinocchio(kin)
    check_closed_form_agrees(kin)
    check_branch_consistency(kin)
    check_determinism(kin)
    check_round_trip(kin)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
