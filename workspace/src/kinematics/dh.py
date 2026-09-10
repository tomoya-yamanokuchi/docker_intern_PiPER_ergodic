#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
The DH parameters AgileX published for the PiPER, as an independent check.

Source: the openrobotics discourse thread "Implementation of forward and inverse
kinematics for AgileX PiPER robotic arm using Eigen library" (topic 50153). Both
conventions are kept because they look like they disagree and do not: standard
row i carries a_i, the offset between joint axes i and i+1, while modified row i
carries a_(i-1), the offset between axes i-1 and i. The same -21.98 mm elbow
offset therefore appears in standard row 3 and modified row 4, and neither is a
wrist parameter -- a4, a5 and d5 are zero in both, which is exactly the
spherical-wrist condition.

These reproduce the URDF to about 0.08 mm; the residual is the 88 um wrist
artefact plus the two-decimal rounding of the theta offsets. Nothing in the
solvers reads this file -- it exists so the model can be checked against a
source that is not the URDF.
"""

import numpy as np

_T2 = np.radians(-172.22)
_T3 = np.radians(-102.78)

# (alpha_i, a_i, d_i, theta_offset_i)
STANDARD = [
    (-np.pi / 2, 0.0, 0.123, 0.0),
    (0.0, 0.28503, 0.0, _T2),
    (np.pi / 2, -0.021984, 0.0, _T3),
    (-np.pi / 2, 0.0, 0.25075, 0.0),
    (np.pi / 2, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.091, 0.0),
]

# (alpha_(i-1), a_(i-1), d_i, theta_i)
MODIFIED = [
    (0.0, 0.0, 0.123, 0.0),
    (-np.pi / 2, 0.0, 0.0, _T2),
    (0.0, 0.28503, 0.0, _T3),
    (np.pi / 2, -0.021984, 0.25075, 0.0),
    (-np.pi / 2, 0.0, 0.0, 0.0),
    (np.pi / 2, 0.0, 0.091, 0.0),
]


def _standard_link(alpha, a, d, theta):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)

    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _modified_link(alpha, a, d, theta):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)

    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def fk_dh(q, convention="standard"):
    """
    Flange pose for the joint angles q [rad], as a 4x4 matrix.
    """
    table, link = (
        (STANDARD, _standard_link) if convention == "standard" else (MODIFIED, _modified_link)
    )

    T = np.eye(4)

    for i, (alpha, a, d, offset) in enumerate(table):
        T = T @ link(alpha, a, d, q[i] + offset)

    return T
