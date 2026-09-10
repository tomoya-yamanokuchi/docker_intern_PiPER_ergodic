#!/usr/bin/env python3
# -*-coding:utf8-*-

import sys
import time


def ensure_can_mode(piper, interface, have_gripper=False, move_spd_rate_ctrl=20, timeout=10.0):
    """
    Report the startup state, and recover it only if it is not already clean.

    A clean exit via mit2can_park() leaves the arm enabled, in CAN control mode,
    MOVE J, holding the zero pose, with 0x151 byte 3 still 0xAD. Expect exactly
    that and touch nothing: the caller's first send_mit() is then the first mode
    frame, and it carries 0xAD too, so the byte never flips.

    Sending a bare ModeCtrl here instead -- which is what enable() does -- flips
    byte 3 to 0x00 with no JointCtrl behind it, and the arm goes limp and stops
    responding until the teaching button is toggled.

    Anything else means a crash, a power-cycle or the teaching button, and goes
    through the recovery handshake, which does drop the arm.
    """
    status = interface.GetArmStatus().arm_status
    enabled = interface.GetArmEnableStatus()

    print(
        "INFO: ctrl_mode:",
        status.ctrl_mode,
        "mode_feed:",
        status.mode_feed,
        "arm_status:",
        status.arm_status,
    )
    print("INFO: Drivers enabled:", enabled)

    if all(enabled) and status.ctrl_mode == 1:
        print("INFO: Arm is already in CAN mode, skipping the handshake.")
        return

    print("INFO: Unexpected startup state, recovering. The arm will drop.")

    if not all(enabled):
        reset_from_mit(piper, interface, have_gripper, move_spd_rate_ctrl)

    teaching2can_mode(piper, interface, timeout, move_spd_rate_ctrl)


def get_q(piper, joints=(1, 2, 3, 4, 5, 6), have_gripper=False):
    """
    Get the current joint angles q [rad] as a {joint number: q} dict.

    Keys are 1-indexed, matching JointMitCtrl's motor_num and the target dicts
    in main.py, so no base[joint-1] shift is needed at the call sites. Pass a
    subset in joints to read only those. With have_gripper the gripper angle is
    appended as joint 7, so ask for it with joints=(1, 2, 3, 4, 5, 6, 7).
    """
    base = piper.get_joint_states()[0]

    if have_gripper:
        base = (*base, piper.get_gripper_states()[0][0])

    return {joint: base[joint - 1] for joint in joints}


def enable(piper, interface, have_gripper=False, move_spd_rate_ctrl=20):
    """
    Enable the robot in normal CAN Joint mode first.
    """
    while not piper.enable_arm():
        time.sleep(0.01)

    if have_gripper:
        piper.enable_gripper()

    # First enter ordinary CAN Joint mode.
    interface.ModeCtrl(
        0x01,  # CAN command control mode
        0x01,  # MOVE J
        move_spd_rate_ctrl,
        0x00,  # normal position/velocity mode
    )

    print("INFO: Enable successful.")


def stop(piper, interface, timeout=10.0):
    """
    Safe reset used when leaving Teaching mode for the first time.

    This follows the same procedure as test_ctrlPiperJoint_can0.py.

    The wait is bounded by timeout because the vendor's safety condition cannot
    always be met: joint 5 has to land inside a one-sided band, so an arm that
    settles near zero there would otherwise spin here forever. Timing out is not
    a new hazard -- EmergencyStop() above has already removed torque, so the arm
    is limp either way and disable_arm() only makes that explicit.
    """
    print("INFO: Emergency stop before switching from Teaching mode.")

    interface.EmergencyStop(0x01)
    time.sleep(1.0)

    # Safety condition copied from the previous working demo. It checks joints
    # 2, 3 and 5 and never joint 4 -- that is the vendor's own condition, kept
    # verbatim. Now written in 1-indexed joint numbers rather than tuple
    # offsets, so the omission is visible; the joints tested are unchanged.
    limit_angle = [0.1745, 0.7854, 0.2094]

    q = get_q(piper)

    print("INFO: Waiting for a safe configuration...")
    print("INFO: Current joints:", q)

    over_time = time.time() + timeout

    while not (
        abs(q[2]) < limit_angle[0]
        and abs(q[3]) < limit_angle[0]
        and q[5] < limit_angle[1]
        and q[5] > limit_angle[2]
    ):
        if over_time < time.time():
            print("ERROR: No safe configuration after", timeout, "s. Disabling anyway at:", q)
            break

        time.sleep(0.01)
        q = get_q(piper)
    else:
        # Reached only when the loop ended without the timeout break above,
        # i.e. the safety condition really was met.
        print("INFO: Safe configuration detected.")

    piper.disable_arm()
    time.sleep(1.0)


def teaching2can_mode(piper, interface, timeout, move_spd_rate_ctrl=20):
    while not piper.enable_arm():
        time.sleep(0.01)

    if interface.GetArmStatus().arm_status.ctrl_mode != 1:
        stop(piper, interface, timeout)

    over_time = time.time() + timeout

    while interface.GetArmStatus().arm_status.ctrl_mode != 1:
        if over_time < time.time():
            print(
                "ERROR: Failed to switch to CAN mode. "
                "Please check whether Teaching mode has been exited."
            )
            sys.exit()

        interface.ModeCtrl(
            0x01,
            0x01,
            move_spd_rate_ctrl,
            0x00,
        )

        time.sleep(0.01)

    enable(piper, interface, False, move_spd_rate_ctrl)


def reset_from_mit(piper, interface, have_gripper=False, move_spd_rate_ctrl=20, timeout=5.0):
    """
    Leave MIT / teaching mode and return to position-velocity CAN control.

    Same sequence as piper_sdk/demo/V2/piper_ctrl_reset.py, which states the
    reset must be run when switching out of MIT or teaching mode. Skipping it
    leaves the arm in MOVE M with its drivers off -- the state that otherwise
    has to be cleared with the physical teaching button.

    piper_ctrl_stop.py notes a reset must be followed by two enables, hence the
    two enable() calls. The arm has no torque between the reset and the first
    enable, so it will sag.
    """
    interface.MotionCtrl_1(0x02, 0, 0)  # 0x02 = resume
    time.sleep(0.1)

    interface.MotionCtrl_2(0, 0, 0, 0x00)  # position/velocity, clears MIT
    time.sleep(0.5)

    enable(piper, interface, have_gripper, move_spd_rate_ctrl)
    enable(piper, interface, have_gripper, move_spd_rate_ctrl)

    # MotionCtrl_2 above left the arm in STANDBY, where it drops its drivers,
    # and the single ModeCtrl inside enable() does not take. Spam it until
    # ctrl_mode reports CAN_CTRL, same as teaching2can_mode() does.
    over_time = time.time() + timeout

    while interface.GetArmStatus().arm_status.ctrl_mode != 1:
        if over_time < time.time():
            print("ERROR: reset_from_mit failed to reach CAN mode.")
            break

        interface.ModeCtrl(0x01, 0x01, move_spd_rate_ctrl, 0x00)
        time.sleep(0.01)
