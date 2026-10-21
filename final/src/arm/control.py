"""What the arm loops share: the rates, the gains, the stdin poll and the two torque laws.

The gains and rates are the live loop's in workspace/, validated on the arm, copied as
numbers. Every torque goes out through arm/piper.py.
"""

import select
import sys
import time
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from arm.piper import apply_joint_torques, read_joint_velocities
from controller.feed_forward import FeedForward
from controller.task_imp_controller import CartesianImpedanceController
from core.agx_pinocchio import AgxPinocchio
from robot import TCP_FRAME, URDF_PATH

CONTROL_FREQ_HZ = 100.0  # the torque
ERGODIC_FREQ_HZ = 20.0  # the law's setpoints, walked between at CONTROL_FREQ_HZ
R_WORLD_BASE = np.eye(3)  # the base is mounted upright


def make_controller() -> CartesianImpedanceController:
    controller = CartesianImpedanceController(
        urdf_path=str(URDF_PATH), dofs=6, frame_name=TCP_FRAME
    )
    controller.set_joint_torque_weights(np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]))
    controller.set_cart_params(
        b=np.array([5.0, 5.0, 5.0, 0.2, 0.2, 0.1]),
        k=np.array([250.0, 250.0, 300.0, 3.0, 3.0, 1.0]),
    )
    return controller


@dataclass
class Impedance:
    """The Cartesian impedance law with the gains validated on the arm, plus friction feedforward."""

    controller: CartesianImpedanceController = field(default_factory=make_controller)
    feed_forward: FeedForward = field(
        default_factory=partial(FeedForward, urdf_path=str(URDF_PATH), dofs=6)
    )

    def send(
        self,
        robot,
        q: np.ndarray,  # (6,) rad, measured this cycle
        target: tuple[np.ndarray, np.ndarray],  # (3,) m and (3, 3), the commanded pose
        twist: np.ndarray,  # (6,) m/s and rad/s, the commanded pose's velocity
    ) -> np.ndarray:  # (6,) N*m, sent
        """The torque towards the commanded pose.

        The feedforward takes only the direction of the desired joint velocity, so a
        pseudo-inverse near a singularity cannot make it large.
        """
        qd = read_joint_velocities(robot)
        tau = self.controller.compute_cartesian_torque(
            desired_pos=target[0],
            desired_ori=target[1],
            q_cur=q,
            v_cur=qd,
            base_orientation=R_WORLD_BASE,
        )
        qd_des = np.linalg.pinv(self.controller.pin_model.jacobian(q, TCP_FRAME)) @ twist
        tau = tau + self.feed_forward.compute_torque(
            q_cur=q, qd_cur=qd, qd_des=qd_des, qdd_des=np.zeros(6)
        )
        apply_joint_torques(robot, tau)
        return tau


def read_command() -> str | None:
    """The line waiting on stdin, stripped, or None.

    Polled, never blocked on: the firmware latches the last torque, so a blocked loop
    would leave a stale one on the arm.
    """
    if not select.select([sys.stdin], [], [], 0)[0]:
        return None
    return sys.stdin.readline().strip()


def drain_stdin() -> None:
    """Discard what was typed earlier, so an old Enter cannot end the next phase at once."""
    while select.select([sys.stdin], [], [], 0)[0] and sys.stdin.readline() != "":
        pass


def keep_rate(start_time: float) -> None:
    """Sleep out the rest of the control period that began at start_time."""
    elapsed, period = time.monotonic() - start_time, 1.0 / CONTROL_FREQ_HZ
    if elapsed < period:
        time.sleep(period - elapsed)
    else:
        print(f"warning: control loop overrun {elapsed:.3f}s > {period:.3f}s")


def gravity_compensation(robot, pin: AgxPinocchio) -> np.ndarray:  # -> (6,) rad, measured
    """One backdrivable cycle: gravity and the velocity terms, nothing else."""
    q = np.array(robot.get_joint_angles().msg)
    qd = read_joint_velocities(robot)
    apply_joint_torques(robot, pin.inverse_dynamics(q, qd, np.zeros(6), R_WORLD_BASE))
    return q
