#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Load the PiPER description once and expose the geometry the solvers need.

Nothing here talks to the arm: this module only reads a URDF.
"""

import os

import numpy as np
import pinocchio as pin

from kinematics.transform import Transform, matrix_to_rpy, rpy_to_matrix

_URDF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "robot_description", "urdf")

# The two variants are kinematically identical and differ only in the inertia of
# gripper_base, link7 and link8. Neither removes the gripper: link7 and link8 are
# present in both, so switching does not move a single frame.
URDF_GRIPPER_MASS = os.path.join(_URDF_DIR, "piper_description_gripper_mass.urdf")
URDF_ZERO_GRIPPER_MASS = os.path.join(_URDF_DIR, "piper_description_zero_gripper_mass.urdf")

# No gripper is fitted at the moment. Leaving its 0.5 kg in the model costs up to
# 3.26 Nm of feedforward error on joint 2 -- 23 times the tau_ff quantisation
# step -- so the bare variant is the default. Switch when a gripper goes back on.
URDF_PATH = URDF_ZERO_GRIPPER_MASS

# The gripper fingers cannot move the flange, so they are locked out of the IK
# vector. Every gripper link and frame survives the reduction.
GRIPPER_JOINTS = ("joint7", "joint8")

# Flange frame. link6 and gripper_base are the same frame in this URDF.
FLANGE_FRAME = "link6"

# Default IK seed and startup pose, in rad.
#
# Not the zero pose: q = 0 sits exactly on the joint 2 and joint 3 limits, and
# j5 = 0 is a wrist singularity, so both are the worst possible seeds. This one
# has sigma_min 0.148 and keeps every joint at least 21% of its range from a
# stop. It is a seed, not a park pose -- the arm falls from here on power loss.
Q_READY = np.array([0.0, 1.70, -1.30, 0.0, 0.70, 0.0])


def _to_transform(placement):
    """
    A pin.SE3 from the URDF parse as this project's own Transform.

    Pinocchio reads the URDF -- it is loaded for rnea anyway, and that keeps the
    URDF the one source of truth -- but nothing downstream should see its types,
    so every placement is converted here at the boundary.
    """
    return Transform(placement.rotation, placement.translation)


def _idealise(M, angle_tol=1e-3, length_tol=1e-4):
    """
    Snap a joint placement to the exact geometry the closed form assumes.

    The URDF writes right angles as 1.5708 rather than pi/2, and carries an
    88 um x-offset on joint 6 that stops the wrist being exactly spherical.
    Both are CAD export artefacts. ik_closed solves this cleaned-up geometry
    and the Newton polish in ik() removes the difference against the real
    model, which is under a micrometre.
    """
    rpy = matrix_to_rpy(M.rotation)
    right_angles = np.round(rpy / (np.pi / 2)) * (np.pi / 2)
    rpy = np.where(np.abs(rpy - right_angles) < angle_tol, right_angles, rpy)

    t = np.where(np.abs(M.translation) < length_tol, 0.0, M.translation)

    return Transform(rpy_to_matrix(*rpy), t)


class PiperKinematics:
    """
    The PiPER model plus the constants its closed-form IK is built on.

    Parses the URDF once; every solver call reuses self.model and self.data.
    tcp_offset [m] shifts the target frame along the flange z axis, which is
    where a peg or a gripper tip goes -- 0.1358 reaches the finger base.
    """

    def __init__(self, urdf_path=URDF_PATH, tcp_offset=0.0):

        # Kept so the meshcat view can read its meshes from the same file this
        # model was built from, rather than from a list of its own.
        self.urdf_path = urdf_path

        self.model_full = pin.buildModelFromUrdf(urdf_path)

        locked = [self.model_full.getJointId(name) for name in GRIPPER_JOINTS]
        self.model = pin.buildReducedModel(self.model_full, locked, pin.neutral(self.model_full))

        # The same TCP offset as self.tcp below, kept on the Pinocchio model so
        # selftest can still check our forward kinematics against Pinocchio's.
        flange = self.model.getFrameId(FLANGE_FRAME)
        placement = pin.SE3(np.eye(3), np.array([0.0, 0.0, tcp_offset]))
        self.frame_id = self.model.addFrame(
            pin.Frame(
                "tcp",
                self.model.frames[flange].parentJoint,
                flange,
                placement,
                pin.FrameType.OP_FRAME,
            )
        )

        self.data = self.model.createData()

        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit

        # The real joint placements, indexed by joint number. fk() walks these
        # and multiplies each by its joint's rotation about local z.
        self.placement = [None] + [
            _to_transform(self.model.jointPlacements[j]) for j in range(1, 7)
        ]

        # The fixed step from the flange to the controlled tool point, which is
        # the last factor in that walk.
        self.tcp = Transform(translation=np.array([0.0, 0.0, tcp_offset]))

        # Idealised placements, indexed by joint number. Only ik_closed uses
        # these; fk() and the Newton polish always use the real geometry.
        self.ideal = [None] + [_idealise(self.placement[j]) for j in range(1, 7)]

        self._set_pieper_constants(tcp_offset)

    def _set_pieper_constants(self, tcp_offset):
        """
        Derive the closed-form geometry from the placements rather than from a
        transcribed DH table, so the numbers cannot drift out of sync with the
        URDF and the other variant still works.
        """
        M1, M2, M3, M4, M6 = (self.ideal[j] for j in (1, 2, 3, 4, 6))

        self.t1 = M1.translation  # base to the joint 1 origin
        self.R2 = M2.rotation
        self.a2 = M3.translation[0]  # upper arm, 0.28503 m
        self.t4 = M4.translation  # forearm vector, in plane
        self.l3 = np.linalg.norm(self.t4)  # forearm length, 0.251712 m
        self.d6 = np.linalg.norm(M6.translation) + tcp_offset

        # Constant angles folded into the planar 2R solve: the yaw baked into
        # M3 and the direction of the offset forearm vector.
        self.q3_offset = np.arctan2(M3.rotation[1, 0], M3.rotation[0, 0])
        self.alpha4 = np.arctan2(self.t4[1], self.t4[0])

        self._check_pieper_assumptions()

    def _check_pieper_assumptions(self):
        """
        Fail loudly if the URDF stops matching the structure ik_closed assumes.

        Without this a changed description would silently produce a solver that
        returns confidently wrong angles.
        """
        M1, M2, M3, M4, M5, M6 = (self.ideal[j] for j in range(1, 7))

        # Each entry is (violated, what it means). Conditions are cheap and
        # side-effect free, so evaluating them all gives the full list of
        # mismatches at once instead of only the first.
        assumptions = (
            (not np.allclose(M1.rotation, np.eye(3)), "joint 1 placement is not axis-aligned"),
            (not np.allclose(M2.translation, 0.0), "joints 1 and 2 do not intersect"),
            (abs(M2.rotation[2, 2]) > 1e-9, "joint 2 axis is not perpendicular to joint 1"),
            (abs(M3.rotation[2, 2] - 1.0) > 1e-9, "joints 2 and 3 are not parallel"),
            (abs(M4.translation[2]) > 1e-9, "joint 4 origin is out of the joint 3 plane"),
            (not np.allclose(M5.translation, 0.0), "joints 4 and 5 do not intersect"),
            (
                abs(M6.translation[0]) > 1e-9,
                "joints 5 and 6 do not intersect -- wrist is not spherical",
            ),
            (not np.allclose(M6.rotation, M5.rotation.T), "wrist rotations are not a ZYZ pair"),
        )

        bad = [reason for violated, reason in assumptions if violated]

        if bad:
            raise ValueError("URDF does not match the Pieper structure -- " + "; ".join(bad))

    def clamp(self, q):
        """
        Clip q into the joint limits.
        """
        return np.clip(q, self.q_min, self.q_max)

    def in_limits(self, q, tol=1e-9):
        """
        True if every joint of q is inside its limit.
        """
        return bool(np.all(q >= self.q_min - tol) and np.all(q <= self.q_max + tol))
