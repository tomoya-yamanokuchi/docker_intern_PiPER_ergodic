#!/usr/bin/env python3
# -*-coding:utf8-*-

import time

from impedance_control.mit import compose_mit_targets, send_mit
from impedance_control.motion import MotionProfile, mit2can_park, move_to
from impedance_control.piper import ensure_can_mode, get_q
from impedance_control.session import connect_arm
from kinematics.dynamics import make_tau_ff_fn
from kinematics.model import PiperKinematics

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
    1: {"qdot_ref": 0.0, "kp": 1.0, "kd": 0.2, "tau_ff": 0.0},  # base yaw, no gravity load
    2: {"qdot_ref": 0.0, "kp": 1.0, "kd": 0.2, "tau_ff": 0.0},  # shoulder, carries the whole arm
    3: {"qdot_ref": 0.0, "kp": 1.0, "kd": 0.2, "tau_ff": 0.0},  # elbow
    4: {"qdot_ref": 0.0, "kp": 1.0, "kd": 0.2, "tau_ff": 0.0},  # forearm roll
    5: {"qdot_ref": 0.0, "kp": 1.0, "kd": 0.2, "tau_ff": 0.0},  # wrist pitch
    6: {"qdot_ref": 0.0, "kp": 1.0, "kd": 0.2, "tau_ff": 0.0},  # wrist roll
}

# How long to let the arm settle on the target before giving up on it.
HOLD_TIME = 10.0

# Fraction of the modelled torque fed forward. Start at 0.0, then raise
# slowly (0.01 -> 0.1 -> 0.3 -> 1.0) watching joint 2, which carries up to
# 8.55 Nm with no gripper fitted. Unlike the kp term, tau_ff is applied
# whether or not there is any position error, so a wrong sign here drives
# the arm rather than merely tracking badly. At 0.01 the largest command is
# 0.09 Nm, below the 0.14 Nm quantisation step, so a mistake is harmless.
TAU_FF_SCALE = 0.0

# Travel speed of the fastest joint [rad/s]. The move is interpolated, so
# this sets how long it takes rather than how hard it is commanded.
MAX_SPEED = 0.3

# Command rate of the streamed trajectory [Hz].
RATE = 100.0

# Used for the normal Joint mode during initialization
MOVE_SPD_RATE_CTRL = 10

# Teaching -> CAN mode timeout
TEACHING_TIMEOUT = 10.0

# No gripper is needed for the peg-in-hole demonstration
HAVE_GRIPPER = False

# Target pose. send_mit() raises on anything outside the joint limits; they are
# tabulated in docs/joint-limits.md, which also covers the gain and torque
# ranges. Note j2 and j3 are one-sided, so the zero pose sits on their limit.
Q_TARGET = {1: 0.0, 2: 1.943, 3: -1.482, 4: 0.0, 5: 1.197, 6: 0.0}

# Parking pose for the finally block. Kept separate from Q_TARGET so the
# exit is always the same however Q_TARGET is edited.
Q_ZERO = {joint: 0.0 for joint in MIT_JOINTS}


def hold_and_report(piper):
    """
    Sample once a second while the arm settles, so the approach and any
    steady-state sag are visible rather than just the end state.

    Reading does not disturb the hold: MIT latches its last setpoint and there
    is no command watchdog.
    """
    held = 0.0

    while held < HOLD_TIME:
        step = min(1.0, HOLD_TIME - held)

        time.sleep(step)
        held += step

        print(f"INFO: t+{held:4.1f}s:", get_q(piper, MIT_JOINTS))

    print("INFO: Reached:", get_q(piper, MIT_JOINTS))


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
    # Model for the feedforward torque. Defaults to the zero-gripper-mass URDF,
    # which is the arm as currently built; pass URDF_GRIPPER_MASS when one is
    # fitted, or the feedforward is wrong by up to 3.26 Nm on joint 2.
    kin = PiperKinematics()

    piper, interface = connect_arm(CAN_PORT)

    print("INFO: Current joints:", get_q(piper, MIT_JOINTS))

    ensure_can_mode(piper, interface, HAVE_GRIPPER, MOVE_SPD_RATE_CTRL, TEACHING_TIMEOUT)
    enter_mit_at_current_pose(piper, interface)

    profile = MotionProfile(
        tuning=TUNING,
        tau_ff_fn=make_tau_ff_fn(kin, TAU_FF_SCALE, MIT_JOINTS),
        max_speed=MAX_SPEED,
        rate=RATE,
    )

    try:
        print("INFO: Moving to", Q_TARGET)

        move_to(piper, interface, Q_TARGET, profile)
        hold_and_report(piper)

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
