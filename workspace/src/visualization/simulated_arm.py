#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
A digital stand-in for the PiPER, so control code can be run with no hardware.

The point is to answer "would this motion do what I meant, and would it hit
anything" on the visualised arm before the same code is pointed at the real one.
So this deliberately implements the *same* surface the SDK objects expose, and
nothing more:

    piper.get_joint_states()        -> ((q1..q6), timestamp, hz)
    piper.get_gripper_states()      -> ((angle, effort), timestamp, hz)
    interface.ModeCtrl(...)
    interface.JointMitCtrl(motor_num, pos_ref, vel_ref, kp, kd, t_ref)
    interface.GetSDKJointLimitParam("j<n>")

One instance stands in for both objects, because move_to() takes them as a pair
and here they are the same fiction. Everything real is still exercised:
compose_mit_targets(), check_targets() and its limit rejection, the smoothstep
trajectory, the feedforward torque model and the telemetry stream. Only the CAN
transport is replaced.

    q_meas == q_cmd

⚠️ Tracking is perfect, which is exactly what the real arm does not do. This
verifies the geometry and the command stream -- reachability, joint limits,
the path taken between waypoints, and whether the arm sweeps through something
it should not. It says nothing about the impedance behaviour: there is no
gravity sag, no lag behind a fast trajectory and no contact, so a gain that is
far too soft looks perfect here.
"""

import time

import numpy as np


class SimulatedArm:
    """
    Records MIT commands and reports them straight back as the measured pose.

    Joint limits come from the model's URDF rather than the SDK's table. The two
    were checked against each other and agree (see docs/joint-limits.md), so
    check_targets() rejects exactly what it would reject on hardware.
    """

    def __init__(self, kin, q_start=None):
        self.kin = kin
        self.q = np.zeros(6) if q_start is None else np.asarray(q_start, dtype=float)

        # Every MIT frame that was "sent", for anyone wanting to assert on the
        # command stream rather than watch it.
        self.commands = []

    def get_joint_states(self):
        return tuple(self.q), time.time(), 0.0

    def get_gripper_states(self):
        return (0.0, 0.0), time.time(), 0.0

    def ModeCtrl(self, ctrl_mode, move_mode, move_spd_rate_ctrl, is_mit_mode):
        self.commands.append(("ModeCtrl", ctrl_mode, move_mode, move_spd_rate_ctrl, is_mit_mode))

    def JointCtrl(self, *angles):
        self.commands.append(("JointCtrl", *angles))
        self.q = np.asarray(angles, dtype=float)

    def JointMitCtrl(self, motor_num, pos_ref, vel_ref, kp, kd, t_ref):
        self.commands.append(("JointMitCtrl", motor_num, pos_ref, vel_ref, kp, kd, t_ref))

        # Perfect tracking: the commanded angle becomes the measured one.
        self.q[motor_num - 1] = pos_ref

    def GetSDKJointLimitParam(self, joint_name):
        joint = int(joint_name.lstrip("j"))

        return self.kin.q_min[joint - 1], self.kin.q_max[joint - 1]

    def GetArmEnableStatus(self):
        return [True] * 6
