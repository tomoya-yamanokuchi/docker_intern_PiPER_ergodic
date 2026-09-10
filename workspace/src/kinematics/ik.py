#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Inverse kinematics: pick the best of the closed-form solutions.

ik_closed solves a slightly idealised arm, so its answers land about 90 um from
the target. That is below the MIT interface's own ~190 um command quantisation,
so it is not worth refining and normally nothing else happens.

The exception is the wrist singularity. Near j5 = 0 the split of a rotation
between j4 and j6 is ill-conditioned, so the same 90 um of geometry error throws
the orientation by up to 5 degrees and every branch misses. Those poses -- 3 in
5000 sampled -- get a short Gauss-Newton repair. The repair exists for that
conditioning problem, not for accuracy.
"""

import numpy as np

from kinematics.fk import fk, jacobian, pose_error
from kinematics.ik_closed import ik_closed
from kinematics.model import Q_READY
from kinematics.transform import log3


def _polish(kin, target, q, iters=30):
    """
    Walk q onto the target with damped Gauss-Newton steps.

    Damping scales with the squared residual so it vanishes as the error does,
    which is what lets this converge in a handful of iterations even though the
    Jacobian is near-singular where it gets used.

    The error is the plain world-frame pair (position difference, rotation
    vector of the residual rotation), paired with the world-aligned Jacobian.
    The earlier version used the SE3 logarithm with its right-Jacobian Jlog6,
    which is more principled but needs two more Pinocchio primitives; measured
    on 79 near-singular poses where the closed form misses, the two repair 75
    and 76 of them respectively, so the simpler one was kept.

    The update is q + v rather than a manifold integration: every joint here is
    revolute with no quaternion in the configuration, so pin.integrate() was
    exactly addition anyway.
    """
    for _ in range(iters):
        current = fk(kin, q)

        error = np.concatenate(
            [
                target.translation - current.translation,
                current.rotation @ log3(current.rotation.T @ target.rotation),
            ]
        )

        if np.linalg.norm(error) < 1e-6:
            break

        J = jacobian(kin, q, local=False)
        damping = 1e-10 + 1e-2 * error.dot(error)

        q = kin.clamp(q + np.linalg.solve(J.T @ J + damping * np.eye(6), J.T @ error))

    return q


def _accept(kin, target, solutions, tol, is_valid):
    """
    Keep the solutions that are in limits, on target, and pass is_valid.

    tol is (position [m], rotation [rad]).
    """
    tol_pos, tol_rot = tol
    keep = []

    for q in solutions:
        position, rotation = pose_error(fk(kin, q), target)

        if position > tol_pos or rotation > tol_rot:
            continue

        if is_valid is not None and not is_valid(q):
            continue

        keep.append(q)

    return keep


def ik(kin, target, q_init=None, tol=(1e-3, 1e-3), is_valid=None, verbose=False):
    """
    Joint angles q [rad] placing the TCP at target, or None if unreachable.

    target is a Transform, the return is a (6,) array always inside the joint
    limits. Of the branches that reach the pose, the one closest to q_init wins,
    so passing the previous solution keeps successive targets on the same elbow
    and wrist configuration instead of reconfiguring mid-path.

    tol is how close counts as reaching it, as (position [m], rotation [rad]).

    is_valid is an optional extra predicate on q, which is where a
    self-collision check goes later.
    """
    if q_init is None:
        q_init = Q_READY

    q_init = np.asarray(q_init, dtype=float)

    # Clamp rather than reject: a branch can land just outside a limit, and near
    # a wrist singularity well outside. _accept throws out any that clamping has
    # actually broken.
    branches = [kin.clamp(q) for q in ik_closed(kin, target, limits_only=False)]

    candidates = _accept(kin, target, branches, tol, is_valid)

    if not candidates:
        candidates = _accept(
            kin, target, (_polish(kin, target, q) for q in branches), tol, is_valid
        )

        if verbose and candidates:
            print("INFO: closed form missed, repaired by polishing")

    if not candidates:
        if verbose:
            print("INFO: no branch reaches this pose")
        return None

    return min(candidates, key=lambda q: np.linalg.norm(q - q_init))
