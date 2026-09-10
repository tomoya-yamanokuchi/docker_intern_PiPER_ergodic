#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
The joint torques a motion needs, for use as MIT feedforward.

Model only -- nothing here talks to the arm.
"""

import numpy as np
import pinocchio as pin


def inverse_dynamics_tau(kin, q, qdot=None, qddot=None):
    """
    Joint torques tau [Nm] the planned motion needs, as a (6,) array.

    With qdot and qddot omitted this is the static hold against gravity, which
    is what a stationary target needs. Passing the planned velocity and
    acceleration adds the inertial and Coriolis terms, which matter once the arm
    is actually moving: at 0.3 rad/s they are worth 0.26 Nm, but they grow with
    the square of the speed and reach 16 Nm at 2 rad/s.

    The sign matches JointMitCtrl's t_ref -- positive is the torque the joint
    must apply. Note this is fed forward regardless of position error, so an
    error here drives the arm rather than merely tracking badly.
    """
    zero = np.zeros(kin.model.nv)

    return pin.rnea(
        kin.model,
        kin.data,
        np.asarray(q, dtype=float),
        zero if qdot is None else np.asarray(qdot, dtype=float),
        zero if qddot is None else np.asarray(qddot, dtype=float),
    )


def make_tau_ff_fn(kin, scale=1.0, joints=(1, 2, 3, 4, 5, 6)):
    """
    Build the tau_ff_fn that move_to() calls, as a {joint: tau} of {joint: q}.

    This is the one place the two index conventions meet: the MIT command path
    is 1-indexed dicts, Pinocchio is 0-indexed arrays. Keeping the conversion
    here means no call site has to write base[joint - 1].

    scale multiplies the whole feedforward, so it can be raised from 0 while
    watching the arm rather than trusted on the first run.
    """

    def tau_ff(q, qdot, qddot):

        tau = inverse_dynamics_tau(
            kin,
            [q[joint] for joint in joints],
            [qdot[joint] for joint in joints],
            [qddot[joint] for joint in joints],
        )

        return {joint: scale * tau[i] for i, joint in enumerate(joints)}

    return tau_ff
