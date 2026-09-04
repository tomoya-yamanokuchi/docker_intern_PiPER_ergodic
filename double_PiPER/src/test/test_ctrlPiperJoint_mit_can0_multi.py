#!/usr/bin/env python3
# -*-coding:utf8-*-

import time
from piper_sdk import Piper


if __name__ == "__main__":

    # ============================================================
    # Configuration
    # ============================================================

    CAN_PORT = "can0"

    # MIT test joint
    # MIT_JOINT = 6
    # MIT_JOINTS = [5, 6]
    # MIT_JOINTS = [4, 5, 6]
    MIT_JOINTS = [3, 4, 5, 6]

    # Small displacement around the CURRENT joint position
    # 0.03 rad ~= 1.72 deg
    # DELTA = 0.03
    DELTA = 0.015

    # MIT impedance parameters
    KP = 10.0
    KD = 0.8
    VEL_REF = 0.0
    TORQUE_FF = 0.0

    # Hold each target for this duration
    HOLD_TIME = 1.0

    # Number of test cycles
    NUM_CYCLES = 3

    # Used for the normal Joint mode during initialization
    move_spd_rate_ctrl = 20

    # Teaching -> CAN mode timeout
    timeout = 10.0

    # No gripper is needed for this test
    have_gripper = False


    # ============================================================
    # Helper functions
    # ============================================================

    def get_pos():
        """
        Get the current joint positions [rad].
        """
        joint_state = piper.get_joint_states()[0]

        if have_gripper:
            return joint_state + (piper.get_gripper_states()[0][0],)

        return joint_state


    def stop():
        """
        Safe reset used when leaving Teaching mode for the first time.

        This follows the same procedure as test_ctrlPiperJoint_can0.py.
        """
        print("INFO: Emergency stop before switching from Teaching mode.")

        interface.EmergencyStop(0x01)
        time.sleep(1.0)

        # Safety condition copied from the previous working demo.
        limit_angle = [0.1745, 0.7854, 0.2094]

        pos = get_pos()

        print("INFO: Waiting for a safe configuration...")
        print("INFO: Current joints:", pos)

        while not (
            abs(pos[1]) < limit_angle[0]
            and abs(pos[2]) < limit_angle[0]
            and pos[4] < limit_angle[1]
            and pos[4] > limit_angle[2]
        ):
            time.sleep(0.01)
            pos = get_pos()

        print("INFO: Safe configuration detected.")

        piper.disable_arm()
        time.sleep(1.0)


    def enable():
        """
        Enable the robot in normal CAN Joint mode first.
        """
        while not piper.enable_arm():
            time.sleep(0.01)

        if have_gripper:
            piper.enable_gripper()

        # First enter ordinary CAN Joint mode.
        interface.ModeCtrl(
            0x01,                  # CAN command control mode
            0x01,                  # MOVE J
            move_spd_rate_ctrl,
            0x00                   # normal position/velocity mode
        )

        print("INFO: Enable successful.")


    def set_mit_mode():
        """
        Switch to MIT control mode.
        """
        interface.ModeCtrl(
            0x01,   # CAN command control mode
            0x04,   # MOVE M / MIT
            0,      # speed percentage is not used here
            0xAD    # MIT mode
        )


    def send_mit(targets):

        set_mit_mode()

        for joint in MIT_JOINTS:
            interface.JointMitCtrl(
                joint,
                targets[joint],
                VEL_REF,
                KP,
                KD,
                TORQUE_FF,
            )


    # ============================================================
    # Connect to PiPER
    # ============================================================

    print("INFO: Connecting to PiPER...")

    piper = Piper(CAN_PORT)

    # Enable SDK-side joint-limit checking.
    # This must be set BEFORE piper.init().
    piper.init_soft_joint_limit_on()

    interface = piper.init()

    piper.connect()

    time.sleep(0.2)

    print("INFO: PiPER connected.")
    print("INFO: Current joints:", get_pos())


    # ============================================================
    # Teaching mode -> CAN mode initialization
    # ============================================================

    while not piper.enable_arm():
        time.sleep(0.01)

    if interface.GetArmStatus().arm_status.ctrl_mode != 1:
        stop()

    over_time = time.time() + timeout

    while interface.GetArmStatus().arm_status.ctrl_mode != 1:

        if over_time < time.time():
            print(
                "ERROR: Failed to switch to CAN mode. "
                "Please check whether Teaching mode has been exited."
            )
            exit()

        interface.ModeCtrl(
            0x01,
            0x01,
            move_spd_rate_ctrl,
            0x00,
        )

        time.sleep(0.01)

    enable()


    # ============================================================
    # Read CURRENT position
    # ============================================================

    time.sleep(0.5)

    # base = list(get_pos())
    # q0 = base[MIT_JOINT - 1]

    base = list(get_pos())

    q0 = {
        joint: base[joint - 1]
        for joint in MIT_JOINTS
    }


    print("")
    print("========================================")
    print("MIT TEST")
    print("========================================")
    # print(f"Joint           : {MIT_JOINT}")
    # print(f"Current position: {q0:.6f} rad")
    print(f"Test amplitude  : +/- {DELTA:.6f} rad")
    print(f"Kp              : {KP}")
    print(f"Kd              : {KD}")
    print("========================================")
    print("")


    # ============================================================
    # IMPORTANT:
    # First enter MIT mode while commanding the CURRENT position.
    #
    # Therefore there should initially be almost no position error.
    # ============================================================

    print("INFO: Entering MIT mode while holding current position...")

    send_mit(q0)

    time.sleep(1.0)


    # ============================================================
    # Small MIT motion around CURRENT position
    #
    # q0
    #  -> q0 + DELTA
    #  -> q0
    #  -> q0 - DELTA
    #  -> q0
    #
    # ============================================================

    try:

        for cycle in range(NUM_CYCLES):

            print("")
            print(
                f"========== Cycle {cycle + 1}/{NUM_CYCLES} =========="
            )

            # targets = [
            #     q0 + DELTA,
            #     q0,
            #     q0 - DELTA,
            #     q0,
            # ]


            # targets_list = [
            #     {
            #         5: q0[5] + DELTA,
            #         6: q0[6] + DELTA,
            #     },
            #     {
            #         5: q0[5],
            #         6: q0[6],
            #     },
            #     {
            #         5: q0[5] - DELTA,
            #         6: q0[6] - DELTA,
            #     },
            #     {
            #         5: q0[5],
            #         6: q0[6],
            #     },
            # ]


            targets_list = [
                {
                    3: q0[3] + DELTA,
                    4: q0[4] + DELTA,
                    5: q0[5] + DELTA,
                    6: q0[6] + DELTA,
                },
                {
                    3: q0[3],
                    4: q0[4],
                    5: q0[5],
                    6: q0[6],
                },
                {
                    3: q0[3] - DELTA,
                    4: q0[4] - DELTA,
                    5: q0[5] - DELTA,
                    6: q0[6] - DELTA,
                },
                {
                    3: q0[3],
                    4: q0[4],
                    5: q0[5],
                    6: q0[6],
                },
            ]


            for targets in targets_list:

                send_mit(targets)

                time.sleep(HOLD_TIME)

                current = get_pos()

                for joint in MIT_JOINTS:
                    print(
                        f"Joint {joint}: "
                        f"target={targets[joint]:.6f}, "
                        f"current={current[joint - 1]:.6f}"
                    )


    except KeyboardInterrupt:

        print("")
        print("INFO: Ctrl+C detected. Stopping MIT test.")


    finally:

        # ========================================================
        # Return to the original center position first.
        # ========================================================

        print("")
        print("INFO: Returning Joint 6 to the original center position.")

        send_mit(q0)

        time.sleep(1.0)

        # ========================================================
        # Switch back to ordinary Joint mode at CURRENT pose.
        # This avoids commanding a new distant target when
        # leaving MIT mode.
        # ========================================================

        current = tuple(get_pos())

        print("INFO: Leaving MIT mode.")
        print("INFO: Holding current joint configuration:", current)

        piper.move_j(
            current,
            move_spd_rate_ctrl
        )

        time.sleep(0.5)

        print("INFO: MIT test finished.")
