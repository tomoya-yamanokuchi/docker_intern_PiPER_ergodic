#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Print this arm's firmware joint limits next to the SDK defaults.

Read-only: it connects and reads feedback, and never commands motion.
The 0x472 query is already sent by connect() via PiperInit().
"""

import math
import time

from piper_sdk import Piper

CAN_PORT = "can0"

piper = Piper(CAN_PORT)
piper.init_soft_joint_limit_on()

interface = piper.init()
piper.connect()

# connect() runs PiperInit() -> SearchAllMotorMaxAngleSpd(); wait for the
# 0x473 replies to come back.
time.sleep(1.0)

motors = interface.GetCurrentMotorAngleLimitMaxVel().all_motor_angle_limit_max_spd.motor

print(f"{'joint':>5} | {'firmware [rad]':>22} | {'SDK default [rad]':>22} | {'max spd':>8}")
print("-" * 70)

for joint in range(1, 7):
    m = motors[joint]

    # max_angle_limit / min_angle_limit are in units of 0.1 degree.
    fw_min = math.radians(m.min_angle_limit * 0.1)
    fw_max = math.radians(m.max_angle_limit * 0.1)

    sdk_min, sdk_max = interface.GetSDKJointLimitParam(f"j{joint}")

    print(
        f"{joint:>5} | {fw_min:>10.4f} {fw_max:>10.4f} | "
        f"{sdk_min:>10.4f} {sdk_max:>10.4f} | {m.max_joint_spd * 0.001:>8.3f}"
    )

print("")
print("Current joints:", piper.get_joint_states()[0])
