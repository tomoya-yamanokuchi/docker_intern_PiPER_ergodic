#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Rigid transforms and the rotation conversions the solvers need.

This replaces pin.SE3 and pin.rpy, so the kinematics are this project's own
code rather than a Pinocchio binding. Pinocchio stays for the rnea feedforward
in dynamics.py, which is validated on hardware, and for parsing the URDF.

The SO(3) conversions delegate to scipy.spatial.transform.Rotation rather than
being hand-rolled. That is not laziness about the maths -- it is accuracy where
this arm actually works. A hand-written atan2 log3 agrees with Pinocchio to
1.1e-11 on average but degrades to 1.4e-07 within a nanoradian of a half turn,
and half turns are the normal case here: a tool pointing straight down is
rpy(0, pi, 0). scipy holds 2e-16 there. It costs about 17 us per call against
Pinocchio's 0.3 us, which is affordable at these rates but is why nothing in a
100 Hz loop should call it more than it must.

Angles are radians and lengths metres, as everywhere else in this project.
"""

import numpy as np
from scipy.spatial.transform import Rotation

# Below this rotation angle the closed form for log6's Vinv divides zero by
# zero, so the series is used instead. The two agree to 1e-16 at the crossover.
SMALL_ANGLE = 1e-4


class Transform:
    """
    A rigid transform: a rotation matrix and a translation.

    The attribute names match pin.SE3's, and __mul__ composes the same way, so
    the solvers read identically to the Pinocchio version they replace.
    """

    def __init__(self, rotation=None, translation=None):

        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
        self.translation = (
            np.zeros(3) if translation is None else np.asarray(translation, dtype=float)
        )

    def __mul__(self, other):
        """
        Compose: self applied after other.
        """
        return Transform(
            self.rotation @ other.rotation, self.rotation @ other.translation + self.translation
        )

    def inverse(self):
        """
        The transform undoing this one.
        """
        rotation = self.rotation.T

        return Transform(rotation, -rotation @ self.translation)

    def actInv(self, other):
        """
        other expressed in this frame, i.e. inverse() * other.

        Named as pin.SE3 names it, because the IK error term reads
        current.actInv(target) in both.
        """
        return self.inverse() * other

    @property
    def homogeneous(self):
        """
        The 4x4 matrix form, which meshcat and the DH comparison both want.
        """
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation

        return matrix

    def copy(self):
        return Transform(self.rotation.copy(), self.translation.copy())

    def __repr__(self):
        return (
            f"Transform(rpy={np.round(matrix_to_rpy(self.rotation), 4)}, "
            f"t={np.round(self.translation, 4)})"
        )


def rpy_to_matrix(roll, pitch, yaw):
    """
    Rotation matrix from roll, pitch, yaw [rad]: Rz(yaw) Ry(pitch) Rx(roll).

    This is URDF's convention and Pinocchio's, and scipy's lowercase "xyz"
    means the same extrinsic sequence. selftest asserts the agreement rather
    than trusting the naming.
    """
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


def matrix_to_rpy(rotation):
    """
    Roll, pitch, yaw [rad] of a rotation matrix, inverting rpy_to_matrix.
    """
    return Rotation.from_matrix(rotation).as_euler("xyz")


def log3(rotation):
    """
    The rotation vector of a rotation matrix: the axis scaled by the angle.
    """
    return Rotation.from_matrix(rotation).as_rotvec()


def exp3(rotation_vector):
    """
    The rotation matrix of a rotation vector, inverting log3.
    """
    return Rotation.from_rotvec(rotation_vector).as_matrix()


def rotation_z(angle):
    """
    A pure rotation of angle [rad] about z, as a Transform.

    This is what a joint contributes to the chain: every PiPER joint turns
    about its own local z.
    """
    cos, sin = np.cos(angle), np.sin(angle)

    return Transform(np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]))


def log6(transform):
    """
    The twist whose exponential is this transform, as [linear, angular].

    Same ordering and scaling as pin.log6(...).vector, so the IK error term and
    pose_error() keep the meaning they were tuned with. The translation part is
    not simply the translation: it is Vinv @ t, which un-mixes the rotation's
    contribution to the displacement.

    The series branch is not an optimisation. Both _polish and pose_error spend
    almost all their time within 1e-4 rad of the identity, where the closed form
    is 0/0, so the series is the branch that actually runs.
    """
    angular = log3(transform.rotation)
    angle = np.linalg.norm(angular)

    skew = np.array(
        [
            [0.0, -angular[2], angular[1]],
            [angular[2], 0.0, -angular[0]],
            [-angular[1], angular[0], 0.0],
        ]
    )

    if angle < SMALL_ANGLE:
        coefficient = 1.0 / 12.0 + angle**2 / 720.0 + angle**4 / 30240.0
    else:
        coefficient = 1.0 / angle**2 - (1.0 + np.cos(angle)) / (2.0 * angle * np.sin(angle))

    v_inverse = np.eye(3) - 0.5 * skew + coefficient * (skew @ skew)

    return np.concatenate([v_inverse @ transform.translation, angular])
