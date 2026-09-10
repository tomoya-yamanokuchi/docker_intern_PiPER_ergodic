#!/usr/bin/env python3
# -*-coding:utf8-*-

import time

import numpy as np
from impedance_control.mit import compose_mit_targets, send_mit
from impedance_control.motion import MotionProfile, mit2can_park, move_to, report_tracking
from impedance_control.piper import ensure_can_mode, get_q
from impedance_control.session import connect_arm
from impedance_control.telemetry import open_telemetry
from kinematics.dynamics import make_tau_ff_fn
from kinematics.ik import ik
from kinematics.model import PiperKinematics
from kinematics.transform import Transform, rpy_to_matrix

CAN_PORT = "can0"
MIT_JOINTS = [1, 2, 3, 4, 5, 6]

# Per-joint MIT tuning. The joint driver runs
#
#     tau = kp * (q - q_meas) + kd * (qdot_ref - qdot_meas) + tau_ff
#
# kp       [Nm/rad]      stiffness. Steady-state sag = gravity torque / kp.
# kd       [Nm/(rad/s)]  damping. With qdot_ref = 0 it brakes all motion.
# qdot_ref [rad/s]       the velocity kd measures against. 0 for a step and
#                        hold; for a streamed trajectory set it to that
#                        joint's desired velocity, otherwise kd drags
#                        against the intended motion and the joint lags.
# tau_ff   [Nm]          feedforward torque added straight to the output.
#                        Put the gravity-hold torque here to remove the sag
#                        without stiffening kp; later, the contact force.
#
# The vendor reference is kp=10, kd=0.8 on every joint; these are far softer.
TUNING = {
    1: {"qdot_ref": 0.0, "kp": 5.0, "kd": 0.8, "tau_ff": 0.0},  # base yaw, no gravity load
    2: {"qdot_ref": 0.0, "kp": 5.0, "kd": 0.8, "tau_ff": 0.0},  # shoulder, carries the whole arm
    3: {"qdot_ref": 0.0, "kp": 5.0, "kd": 0.8, "tau_ff": 0.0},  # elbow
    4: {"qdot_ref": 0.0, "kp": 1.0, "kd": 0.25, "tau_ff": 0.0},  # forearm roll
    5: {"qdot_ref": 0.0, "kp": 1.0, "kd": 0.25, "tau_ff": 0.0},  # wrist pitch
    6: {"qdot_ref": 0.0, "kp": 1.0, "kd": 0.1, "tau_ff": 0.0},  # wrist roll
}

# How long to let the arm settle on the target before giving up on it.
HOLD_TIME = 1.0

# Fraction of the modelled torque fed forward. Start at 0.0, then raise
# slowly (0.01 -> 0.1 -> 0.3 -> 1.0) watching joint 2, which carries up to
# 8.55 Nm with no gripper fitted. Unlike the kp term, tau_ff is applied
# whether or not there is any position error, so a wrong sign here drives
# the arm rather than merely tracking badly. At 0.01 the largest command is
# 0.09 Nm, below the 0.14 Nm quantisation step, so a mistake is harmless.
TAU_FF_SCALE = 1.0

# Travel speed of the fastest joint [rad/s]. The move is interpolated, so
# this sets how long it takes rather than how hard it is commanded.
MAX_SPEED = 2.0

# Command rate of the streamed trajectory [Hz].
RATE = 100.0

# Distance from the flange to the controlled tool point, along the flange
# z axis [m]. IK targets are this frame, not the flange.
TCP_OFFSET = 0.06

# Used for the normal Joint mode during initialization
MOVE_SPD_RATE_CTRL = 10

# Teaching -> CAN mode timeout
TEACHING_TIMEOUT = 10.0

# No gripper is needed for the peg-in-hole demonstration
HAVE_GRIPPER = False

# Target pose. send_mit() raises on anything outside the joint limits; they are
# tabulated in docs/joint-limits.md, which also covers the gain and torque
# ranges. Note j2 and j3 are one-sided, so the zero pose sits on their limit.
Q_TARGET = {1: 0.0, 2: 1.512, 3: -0.902, 4: 0.0, 5: 1.048, 6: 0.0}

# Parking pose for the finally block. Kept separate from Q_TARGET so the
# exit is always the same however Q_TARGET is edited.
Q_ZERO = {joint: 0.0 for joint in MIT_JOINTS}

# The same pose again, but asked for in Cartesian terms and solved by IK
# rather than written out by hand. TCP_OFFSET moves the controlled frame
# down the flange z axis, so the target below is where the tool tip goes,
# not the flange.
TCP_TARGET = Transform(
    rpy_to_matrix(0.0, 3.1416, 0.0),  # tool pointing down
    np.array([0.2, 0.1, 0.15]),
)  # metres


def solve_tcp_target(kin):
    """
    Joint angles reaching TCP_TARGET, as a 1-indexed {joint: q} dict.
    """
    q_solution = ik(kin, TCP_TARGET)

    if q_solution is None:
        raise ValueError(f"No IK solution for {TCP_TARGET.translation}")

    # ik() returns a 0-indexed array; the MIT command path is 1-indexed.
    q_ik = {joint: q_solution[joint - 1] for joint in MIT_JOINTS}

    print(
        "INFO: IK solved",
        np.round(TCP_TARGET.translation, 4),
        "->",
        {joint: round(q, 4) for joint, q in q_ik.items()},
    )

    return q_ik


def hold_and_report(piper, name, target):
    """
    Sample once a second while the arm settles on target.

    The approach and any steady-state sag are then visible rather than just the
    end state. Reading does not disturb the hold: MIT latches its last setpoint
    and there is no command watchdog.
    """
    held = 0.0

    while held < HOLD_TIME:
        step = min(1.0, HOLD_TIME - held)

        time.sleep(step)
        held += step

        report_tracking(
            f"holding {name}, t+{held:4.1f}s", target, get_q(piper, MIT_JOINTS), MIT_JOINTS
        )

    report_tracking(f"settled at {name}", target, get_q(piper, MIT_JOINTS), MIT_JOINTS)


def enter_mit_at_current_pose(piper, interface):
    """
    Switch into MIT at the pose the arm is already holding, so the position
    error at the switch is near zero and the arm does not jump.
    """
    time.sleep(0.5)

    q0 = get_q(piper, MIT_JOINTS)

    print("INFO: Entering MIT mode at", q0)

    send_mit(interface, compose_mit_targets(q0, TUNING))

    time.sleep(1.0)


def main():
    # Model for IK and for the feedforward torque. Defaults to the
    # zero-gripper-mass URDF, which is the arm as currently built; pass
    # URDF_GRIPPER_MASS when one is fitted, or the feedforward is wrong by up
    # to 3.26 Nm on joint 2. tcp_offset only moves the frame IK targets, so the
    # same model serves both jobs.
    kin = PiperKinematics(tcp_offset=TCP_OFFSET)

    piper, interface = connect_arm(CAN_PORT)

    print("INFO: Current joints:", get_q(piper, MIT_JOINTS))

    ensure_can_mode(piper, interface, HAVE_GRIPPER, MOVE_SPD_RATE_CTRL, TEACHING_TIMEOUT)
    enter_mit_at_current_pose(piper, interface)

    q_ik = solve_tcp_target(kin)

    # The demo: establish a known start at zero, out to the hand-written joint
    # target, back to zero, then out again to the IK solution.
    demo = [
        ("the zero pose", Q_ZERO),
        ("the joint-space target", Q_TARGET),
        ("the zero pose", Q_ZERO),
        ("the IK solution", q_ik),
        ("the zero pose", Q_ZERO),
    ]

    # The feedforward depends only on the model and the scale, so it is built
    # once and reused by every move, the parking travel included. Telemetry is
    # always on: an unread UDP datagram is dropped by the kernel, so this costs
    # one sendto per step whether or not the visualization application is running, and a
    # viewer can be attached or killed mid-run without touching the arm.
    profile = MotionProfile(
        tuning=TUNING,
        tau_ff_fn=make_tau_ff_fn(kin, TAU_FF_SCALE, MIT_JOINTS),
        max_speed=MAX_SPEED,
        rate=RATE,
        telemetry=open_telemetry(),
    )

    try:
        for name, target in demo:
            print("")
            print(f"INFO: Moving to {name}:", target)

            move_to(piper, interface, target, profile)
            hold_and_report(piper, name, target)

    except KeyboardInterrupt:
        print("")
        print("INFO: Ctrl+C detected. Stopping program")

    finally:
        print("INFO: Parking at the zero pose and handing back to CAN mode.")

        mit2can_park(
            piper,
            interface,
            Q_ZERO,
            profile,
            hold_time=1.0,
            move_spd_rate_ctrl=MOVE_SPD_RATE_CTRL,
        )


if __name__ == "__main__":
    main()
