#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Closed-form inverse kinematics by Pieper decoupling.

The PiPER's joint 4, 5 and 6 axes meet at a point, so the wrist centre depends
only on q1, q2, q3 and the remaining orientation only on q4, q5, q6. That splits
the problem into a planar 2R solve and a ZYZ Euler extraction, and gives up to
eight exact solutions instead of the one basin a seeded Newton iteration finds.

These solve the idealised geometry from model._idealise(); the caller polishes
the sub-micrometre remainder against the real model.
"""

import numpy as np

from kinematics.transform import rotation_z

SINGULAR_TOL = 1e-9

# Slack on the two cosines that decide reachability. The idealised geometry is
# 88 um from the real one, so a pose right on the workspace boundary lands just
# outside the valid domain and would be rejected as unreachable. This is worth
# about 0.7 mm of over-reach; the Newton polish pulls the result back on.
REACH_TOL = 1e-2

# Slack on the joint limits. Same reason: a solution 88 um from a limit may
# come back a hair outside it. ik() clamps and re-checks strictly after the
# polish, so nothing out of limits escapes.
LIMIT_TOL = 1e-2


def _wrap(q):
    """
    Wrap angles into [-pi, pi]. Every PiPER joint range fits inside it.
    """
    return (q + np.pi) % (2.0 * np.pi) - np.pi


def wrist_centre(kin, target):
    """
    World position of the point where the joint 4, 5 and 6 axes meet.

    It sits d6 back along the tool z axis, so it is fixed by the target pose
    alone and needs no joint angles.
    """
    return target.translation - kin.d6 * target.rotation[:, 2]


def _solve_q1(kin, p_w):
    """
    The two base rotations that bring the wrist centre into the joint 2 plane.

    Joint 2 sweeps a plane whose normal is its own axis, so q1 is fixed by
    requiring the wrist centre to lie in that plane: A cos(q1) + B sin(q1) + C = 0.
    """
    u = p_w - kin.t1
    n = kin.R2[:, 2]

    A = n[0] * u[0] + n[1] * u[1]
    B = n[0] * u[1] - n[1] * u[0]
    C = n[2] * u[2]

    radius = np.hypot(A, B)

    if radius < SINGULAR_TOL:
        return []  # wrist centre on the joint 1 axis, q1 is free

    if abs(C) > radius * (1.0 + REACH_TOL):
        return []

    phi = np.arctan2(B, A)
    delta = np.arccos(np.clip(-C / radius, -1.0, 1.0))

    return [phi + delta, phi - delta]


def _solve_q2_q3(kin, p_w, q1):
    """
    The two elbow configurations reaching the wrist centre for a given q1.

    In the joint 2 plane this is a 2R arm of length a2 and l3, except that the
    forearm vector is not aligned with its own link -- alpha4 is that constant
    bend, and q3_offset the yaw already baked into the joint 3 placement.
    """
    w = kin.R2.T @ (rotation_z(-q1).rotation @ (p_w - kin.t1))

    L = np.hypot(w[0], w[1])

    cos_elbow = (L * L - kin.a2**2 - kin.l3**2) / (2.0 * kin.a2 * kin.l3)

    if abs(cos_elbow) > 1.0 + REACH_TOL:
        return []  # too far away, or inside the fold

    elbow = np.arccos(np.clip(cos_elbow, -1.0, 1.0))

    out = []

    for theta3 in (elbow - kin.alpha4, -elbow - kin.alpha4):
        reach = np.array([kin.a2, 0.0, 0.0]) + rotation_z(theta3).rotation @ kin.t4

        q2 = np.arctan2(w[1], w[0]) - np.arctan2(reach[1], reach[0])

        out.append((q2, theta3 - kin.q3_offset))

    return out


def _solve_wrist(kin, target, q1, q2, q3):
    """
    The two wrist configurations giving the target orientation.

    Because joints 4, 5, 6 are concurrent and the joint 5 and 6 placements are
    an exact inverse pair, the residual rotation is a plain ZYZ Euler triple.
    """
    oM3 = (
        kin.ideal[1]
        * rotation_z(q1)
        * kin.ideal[2]
        * rotation_z(q2)
        * kin.ideal[3]
        * rotation_z(q3)
    )

    W = (oM3.rotation @ kin.ideal[4].rotation).T @ target.rotation

    sin_q5 = np.hypot(W[0, 2], W[1, 2])

    if sin_q5 < SINGULAR_TOL:
        # Wrist singularity: q4 and q6 only appear as a sum, so pin q4 at zero
        # and let the Newton polish move off the degeneracy if it needs to.
        if W[2, 2] > 0.0:
            return [(0.0, 0.0, np.arctan2(W[1, 0], W[0, 0]))]
        return [(0.0, np.pi, -np.arctan2(-W[1, 0], -W[0, 0]))]

    q5 = np.arctan2(sin_q5, W[2, 2])
    q4 = np.arctan2(W[1, 2], W[0, 2])
    q6 = np.arctan2(W[2, 1], -W[2, 0])

    return [(q4, q5, q6), (q4 + np.pi, -q5, q6 + np.pi)]


def ik_closed(kin, target, limits_only=True):
    """
    Every closed-form solution for the target pose, as a list of (6,) arrays.

    Up to eight: two base rotations x two elbow bends x two wrist flips. With
    limits_only the ones outside the joint limits are dropped, which is usually
    most of them -- joints 2 and 3 have one-sided ranges.
    """
    p_w = wrist_centre(kin, target)

    solutions = []

    for q1 in _solve_q1(kin, p_w):
        for q2, q3 in _solve_q2_q3(kin, p_w, q1):
            for q4, q5, q6 in _solve_wrist(kin, target, q1, q2, q3):
                q = _wrap(np.array([q1, q2, q3, q4, q5, q6]))

                if not limits_only or kin.in_limits(q, tol=LIMIT_TOL):
                    solutions.append(q)

    return solutions
