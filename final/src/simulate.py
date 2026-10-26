#!/usr/bin/env python3
"""The whole chain offline, on the fake arm: record a master, teach, label, run.

Hardware-free. A synthetic locked-door master (insert the key INSERT along its axis, turn
it TURN about that axis, push PUSH further) is made by IK and played through
record_master.py, as if backdriven. TEACH datapoints scattered around it are played
through teach_exploration.py with an Enter on each. label.py labels them, and
run_trials.py runs the method on the driven fake arm, in real time, watchable in MeshCat.

The fake arm has no door, so nothing blocks a stage and nothing needs a retry; the scripted
operator places the arm at the master's start and then keeps its hands off, so a trial ends
as run_trials ends one on the arm. What this proves is the chain and the loop, not the
method's success on the task. Files go to final/output/sim.

The fake arm is idealised, with no friction, to check the method and the loop. The
feedforward is set as it would be identified on such an arm: no friction term and no
gravity correction (the one measured on the real arm's joint 2). run_trials.py is the
same as on the arm.

Run from final/src:  sh run.sh simulate.py [e2t2|sliding|retry] [n_trials]
"""

import sys
from itertools import pairwise

import numpy as np
from scipy.spatial.transform import Rotation as R

import label
import record_master
import run_trials
import teach_exploration
from arm.fake import FakePiper
from core.agx_pinocchio import AgxPinocchio
from method.pose import ROTATION_LENGTH
from robot import OUTPUT_DIR, TCP_FRAME, URDF_PATH
from visualization.views import TeachingView

OUT = OUTPUT_DIR / "sim"
START = np.array([0.30, 0.0, 0.25])  # m, the key's tip before insertion
KEY_FORWARD = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # key axis along +x
INSERT, TURN, PUSH = 0.04, np.pi / 2, 0.05  # m, rad, m
SAMPLES, DWELL = 600, 50  # the master's motion and the stand at either end, in 100 Hz samples
# Datapoints, and their scatter in the metric per axis: about the E2T2 paper's own spread
# of separate insertions, 15 mm and 10 degrees.
TEACH, SCATTER = 200, 0.008  # m
MOVE = 10  # cycles the simulated operator takes from one datapoint to the next
SEED_Q = np.array([0.0, 1.2, -1.0, 0.0, 0.6, 0.0])  # rad, where the first IK starts


def ik(pin: AgxPinocchio, q: np.ndarray, p: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Damped least squares from q to the tool pose (p, rot); (6,) rad."""
    for _ in range(200):
        p_i, r_i = pin.forward_kinematics(q, TCP_FRAME)
        err = np.concatenate([p - p_i, R.from_matrix(rot @ r_i.T).as_rotvec()])
        if np.linalg.norm(err) < 1e-8:
            break
        J = pin.jacobian(q, TCP_FRAME)
        q = q + J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), err)
    return q


def master_poses() -> tuple[np.ndarray, np.ndarray]:
    """The synthetic master's tool positions (n, 3) m and orientations (n, 3, 3)."""
    u = np.concatenate([np.zeros(DWELL), np.linspace(0, 3, SAMPLES), np.full(DWELL, 3.0)])
    insert, turn, push = np.clip(u, 0, 1), np.clip(u - 1, 0, 1), np.clip(u - 2, 0, 1)
    p = START + np.outer(INSERT * insert + PUSH * push, KEY_FORWARD[:, 2])
    rot = KEY_FORWARD @ R.from_rotvec(np.outer(TURN * turn, [0, 0, 1])).as_matrix()
    return p, rot


def joint_path(pin: AgxPinocchio, p: np.ndarray, rot: np.ndarray) -> np.ndarray:  # (n, 6) rad
    q, out = SEED_Q, []
    for p_i, r_i in zip(p, rot, strict=True):
        q = ik(pin, q, p_i, r_i)
        out.append(q)
    return np.array(out)


def scattered(
    pin: AgxPinocchio, p: np.ndarray, rot: np.ndarray, q_path: np.ndarray, rng
) -> np.ndarray:
    """TEACH joint poses near random master samples, scattered by SCATTER in the metric."""
    pick = np.sort(rng.integers(DWELL, len(p) - DWELL, TEACH))  # visited along the master
    dp = rng.normal(0.0, SCATTER, (TEACH, 3))
    dr = R.from_rotvec(rng.normal(0.0, SCATTER / ROTATION_LENGTH, (TEACH, 3))).as_matrix()
    return np.array([ik(pin, q_path[k], p[k] + dp[i], dr[i] @ rot[k]) for i, k in enumerate(pick)])


def teaching_path(q_points: np.ndarray) -> tuple[np.ndarray, set[int]]:
    """An operator's path through the datapoints, MOVE cycles between each, and the cycles to confirm on."""
    path, confirm = [q_points[0]], set()
    for a, b in pairwise(q_points):
        confirm.add(len(path) - 1)
        path.extend(np.linspace(a, b, MOVE + 1)[1:])
    confirm.add(len(path) - 1)
    path.append(q_points[-1])  # one cycle more, so the last confirmation is read
    return np.array(path), confirm


class Operator:
    """Stands in for the person at the arm during run_trials: places it, then never aborts."""

    def __init__(self, robot: FakePiper, task):
        self.robot, self.task, self.placing = robot, task, False
        self.rng = np.random.default_rng(1)

    def drain(self) -> None:  # run_trials calls it as placing begins
        self.placing = True

    def read(self) -> str | None:
        if self.placing:
            self.placing = False
            self.robot.q = self.task.q_master[0] + self.rng.normal(0.0, 0.01, 6)
            self.robot.qd[:] = 0.0
            return ""
        return None


def main(method: str, n_trials: int) -> None:
    pin, rng = AgxPinocchio(str(URDF_PATH)), np.random.default_rng(0)
    p, rot = master_poses()
    q_path = joint_path(pin, p, rot)
    p_back, _ = pin.forward_kinematics(q_path[-1], TCP_FRAME)
    print(
        f"synthetic master: {len(q_path)} samples, IK end error {np.linalg.norm(p_back - p[-1]):.1e} m"
    )
    master = record_master.main(FakePiper(q_path[0], path=q_path, max_cycles=len(q_path)), OUT)

    q_teach, confirm = teaching_path(scattered(pin, p, rot, q_path, rng))
    robot = FakePiper(q_teach[0], path=q_teach, max_cycles=len(q_teach))
    # The q a cycle reads is the path's sample before the cycle's torque advances it.
    teach_exploration.read_command = lambda: "" if robot.cycles - 1 in confirm else None
    exploration = teach_exploration.main(robot, TeachingView(np.load(master)["q"]), None, OUT)

    task_path = label.main(master, exploration, OUT)
    setup = run_trials.prepare(method, task_path)
    robot = FakePiper(q_path[0])
    ideal = setup[1].feed_forward
    ideal.set_friction_params(*[np.zeros(6)] * 3, stribeck_velocity=np.ones(6))
    ideal.gravity_scale = np.ones(6)
    operator = Operator(robot, setup[4])
    run_trials.read_command, run_trials.drain_stdin = operator.read, operator.drain
    run_trials.main(robot, setup, method, task_path, n_trials, OUT)


if __name__ == "__main__":
    method, n_trials = sys.argv[1:3] + ["retry", "1"][len(sys.argv[1:3]) :]
    main(method, int(n_trials))
