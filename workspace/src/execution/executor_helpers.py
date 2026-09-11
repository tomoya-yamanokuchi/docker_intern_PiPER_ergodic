"""Arm I/O shared by the execution scripts: connect, read, command, hold."""

import contextlib
import time

import numpy as np
from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config


def connect_arm():
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=PiperFW.DEFAULT,  # this arm reports S-V1.8-2
        channel="can0",
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()
    while not robot.enable():
        time.sleep(1)
    print("arm enabled")
    # get_joint_angles() is None until the first frame arrives.
    while robot.get_joint_angles() is None:
        time.sleep(0.01)
    return robot


def read_joint_velocities(robot) -> np.ndarray:  # (6,) rad/s
    joint_velocities = np.zeros(robot.joint_nums)
    for joint_id in range(1, robot.joint_nums + 1):
        ms = robot.get_motor_states(joint_id)
        if ms is not None:
            joint_velocities[joint_id - 1] = ms.msg.velocity
    return joint_velocities


def apply_joint_torques(robot, tau: np.ndarray) -> None:  # tau (6,) N·m
    try:
        for joint_id in range(1, robot.joint_nums + 1):
            robot.move_mit(joint_id, 0, 0, 0, 0, tau[joint_id - 1])
    except Exception as e:
        print(f"applying joint torques failed: {e}")


def hold_current_pose(robot, joint_angles: np.ndarray) -> None:  # joint_angles (6,) rad
    # The arm has no brakes: hand it to the joint driver's own PD loop. Each
    # joint is tried even if another fails.
    for joint_id in range(1, robot.joint_nums + 1):
        with contextlib.suppress(Exception):
            robot.move_mit(joint_id, joint_angles[joint_id - 1], 0, 10, 0.8, 0)
