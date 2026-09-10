#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Opening a CAN session with a real PiPER.

This is deliberately the only module in impedance_control/ that imports
piper_sdk. Everything else -- get_q(), send_mit(), move_to() -- is handed the
handles and never constructs one, which is what lets the same control code run
against visualization/simulated_arm.py with no SDK, no CAN and no arm.

Importing this module is therefore the line between "offline" and "on the
hardware": if a tool needs it, it needs a robot.
"""

import time

from piper_sdk import Piper


def connect_arm(can_port="can0"):
    """
    Open the CAN session and return (piper, interface).

    init_soft_joint_limit_on() must run before init(), which snapshots the
    flags; connect() starts the read thread, without which get_joint_states()
    silently returns zeros. The short sleep lets the first feedback frames
    arrive before anything reads a pose.
    """
    print("INFO: Connecting to PiPER...")

    piper = Piper(can_port)
    piper.init_soft_joint_limit_on()

    interface = piper.init()
    piper.connect()

    time.sleep(0.2)

    return piper, interface
