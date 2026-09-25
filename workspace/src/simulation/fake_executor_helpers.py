"""A stand-in for execution.executor_helpers: the same four functions, no pyAgxArm, no CAN.

The fake arm sits still at a fixed q and records the torques it is sent, so an
execution script's main() can run offline:

    fake_executor_helpers.install(q, max_cycles=300)
    import run_ergodic_pipeline  # now gets the fake helpers as its SDK boundary
"""

import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace

import numpy as np


@dataclass
class FakeArm:
    q: np.ndarray  # (6,) rad, never changes
    # apply_joint_torques raises KeyboardInterrupt on this call, as Ctrl-C would.
    max_cycles: int | None = None
    torques: list[np.ndarray] = field(default_factory=list)  # (6,) N·m per call
    held_pose: np.ndarray | None = None  # (6,) rad, set by hold_current_pose
    joint_nums: int = 6

    def get_joint_angles(self) -> SimpleNamespace:
        return SimpleNamespace(msg=list(self.q))


def install(q: np.ndarray, max_cycles: int | None = None) -> FakeArm:  # q (6,) rad
    """Register a module as execution.executor_helpers; import the script afterwards."""
    arm = FakeArm(np.asarray(q, dtype=float), max_cycles)
    helpers = ModuleType("execution.executor_helpers")
    helpers.connect_arm = lambda: arm
    helpers.read_joint_velocities = read_joint_velocities
    helpers.apply_joint_torques = apply_joint_torques
    helpers.hold_current_pose = hold_current_pose
    sys.modules[helpers.__name__] = helpers
    return arm


def read_joint_velocities(robot: FakeArm) -> np.ndarray:  # (6,) rad/s
    return np.zeros(robot.joint_nums)


def apply_joint_torques(robot: FakeArm, tau: np.ndarray) -> None:  # tau (6,) N·m
    robot.torques.append(np.asarray(tau, dtype=float))
    if len(robot.torques) == robot.max_cycles:
        raise KeyboardInterrupt


def hold_current_pose(robot: FakeArm, joint_angles: np.ndarray) -> None:  # (6,) rad
    robot.held_pose = np.asarray(joint_angles, dtype=float)
