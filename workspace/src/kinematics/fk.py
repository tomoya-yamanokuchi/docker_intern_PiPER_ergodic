#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Forward kinematics and the frame Jacobian, in numpy.

Every PiPER joint turns about its own local z (<axis xyz="0 0 1"/> on joints 1
to 6), so the whole chain is one product repeated: the fixed placement of each
joint, then its own rotation. That makes hand-written forward kinematics a few
lines, which is why this no longer calls Pinocchio. Pinocchio still parses the
URDF that supplies those placements, and still does the rnea feedforward.
"""

import numpy as np

from kinematics.transform import Transform, log6, rotation_z


def joint_frames(kin, q):
    """
    The six joint frames followed by the TCP frame, as a list of Transform.

    A link's frame is the frame of the joint that drives it, so this is also
    what the meshcat view hangs each mesh on. Each joint rotates about the z
    axis of its own frame, which is what the visualiser draws through the wrist.
    """
    q = np.asarray(q, dtype=float)

    frames = []
    frame = Transform()

    for joint in range(1, 7):
        frame = frame * kin.placement[joint] * rotation_z(q[joint - 1])
        frames.append(frame)

    frames.append(frame * kin.tcp)

    return frames


def fk(kin, q):
    """
    Pose of the TCP frame for the joint angles q [rad], as a Transform.
    """
    return joint_frames(kin, q)[-1]


def jacobian(kin, q, local=True):
    """
    Frame Jacobian of the TCP, as a (6, 6) array.

    Column i is what joint i alone does to the tool. A revolute joint turning
    about the world axis z_i through p_i carries the tip at z_i x (p_tcp - p_i)
    and rotates it at z_i -- that is the whole derivation, no differentiation
    needed.

    local expresses the result in the TCP frame rather than in world axes, and
    is the default because that is the frame the IK error is measured in.
    """
    frames = joint_frames(kin, q)
    tip = frames[-1].translation

    J = np.zeros((6, 6))

    for i, frame in enumerate(frames[:-1]):
        axis = frame.rotation[:, 2]

        J[:3, i] = np.cross(axis, tip - frame.translation)
        J[3:, i] = axis

    if local:
        rotation = frames[-1].rotation.T
        J = np.vstack([rotation @ J[:3], rotation @ J[3:]])

    return J


def pose_error(current, target):
    """
    Position [m] and orientation [rad] error magnitudes between two poses.
    """
    error = log6(current.actInv(target))

    return np.linalg.norm(error[:3]), np.linalg.norm(error[3:])
