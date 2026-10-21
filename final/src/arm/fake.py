"""A simulated PiPER behind the pyAgxArm surface arm/piper.py uses, so any script runs offline.

Hand a FakePiper to a script's main() in place of connect_arm(). Every move_mit is the
driver's own law, kp (p - q) + kd (v - qd) + t_ff; the call for joint 6 ends a control
cycle, and the arm then moves one control period under it:

- driven: an idealised arm, the URDF's rigid-body dynamics with no friction, plus the
  motors' rotor inertia as a stand-in (ARMATURE). It checks the method and the loop, not
  how the real arm tracks. There is nothing to touch.
- played: q follows a given path one sample per cycle whatever the torque, standing in
  for an operator backdriving the arm.

A KeyboardInterrupt is raised once, at max_cycles, as a Ctrl-C would be.
"""

from types import SimpleNamespace

import numpy as np
import pinocchio as pin

from core.agx_pinocchio import AgxPinocchio
from robot import URDF_PATH

PERIOD = 0.01  # s, one control cycle at 100 Hz
SUBSTEPS = 10  # the torque is held over a cycle, as the driver holds it; the arm moves at 1 kHz
# kg*m^2 per joint: the motors' reflected rotor inertia, which the URDF leaves out. Not a
# measured value, only its order: with the URDF's wrist links alone (1e-4) the sampled
# wrist rings up within five cycles.
ARMATURE = 0.03


class FakePiper:
    joint_nums = 6

    def __init__(
        self,
        q: np.ndarray,
        path: np.ndarray | None = None,
        max_cycles: int | None = None,
    ):
        self.model = AgxPinocchio(str(URDF_PATH)).robot.model
        self.model.armature[:] = ARMATURE
        self.data = self.model.createData()
        self.q, self.qd = np.array(q, float), np.zeros(6)
        self.path, self.max_cycles = path, max_cycles
        self.tau, self.cycles = np.zeros(6), 0

    def get_joint_angles(self) -> SimpleNamespace:
        return SimpleNamespace(msg=list(self.q))

    def get_motor_states(self, joint_id: int) -> SimpleNamespace:
        return SimpleNamespace(msg=SimpleNamespace(velocity=self.qd[joint_id - 1]))

    def get_config(self) -> dict:
        return {
            "joint_torque_b": (4, 4, 4, 1, 1, 1),
            "joint_torque_c": (1, 1, 1, 0.813252, 0.813252, 0.813252),
        }

    def move_mit(self, joint_id: int, p_des, v_des, kp, kd, t_ff) -> None:
        j = joint_id - 1
        self.tau[j] = kp * (p_des - self.q[j]) + kd * (v_des - self.qd[j]) + t_ff
        if joint_id == self.joint_nums:
            self.cycles += 1
            if self.cycles == self.max_cycles:
                raise KeyboardInterrupt
            self._play() if self.path is not None else self._drive()

    def _play(self) -> None:
        q = self.path[min(self.cycles, len(self.path) - 1)]
        self.q, self.qd = q.copy(), (q - self.q) / PERIOD

    def _drive(self) -> None:
        dt = PERIOD / SUBSTEPS
        lo, hi = self.model.lowerPositionLimit, self.model.upperPositionLimit
        for _ in range(SUBSTEPS):
            self.qd = self.qd + dt * pin.aba(self.model, self.data, self.q, self.qd, self.tau)
            self.q = self.q + dt * self.qd
            stopped = (self.q < lo) | (self.q > hi)
            self.q, self.qd[stopped] = np.clip(self.q, lo, hi), 0.0
