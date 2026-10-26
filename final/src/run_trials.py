#!/usr/bin/env python3
"""Run the method on the arm: N trials of the locked-door task, one method throughout.

TOUCHES THE ARM. Methods: e2t2 (static target, its statistic handed on across trials, as
its paper does), sliding (sliding target with forgetting, fresh each trial) and retry
(sliding with the retry). Before every trial the arm is backdrivable: re-lock the door,
take the key out, put the arm near the master's start (the distance is printed) and press
Enter. The law is not stepped meanwhile, so the handling never enters the statistic.

During a trial, Enter ends it as a success and "a" as a failure; so does the time limit,
TIME_LIMIT. Ctrl-C holds the arm and saves the trials so far
to run_<method>_<timestamp>.npz in final/output; view one with view_run.py.

Control is fully online, as closed-loop ergodic control is meant to be: every
ERGODIC_FREQ_HZ the measured pose goes into the law and its next state comes out,
MAX_SPEED / ERGODIC_FREQ_HZ away plus whatever the arm left of the last step, and the
commanded pose walks from the measured pose to it at CONTROL_FREQ_HZ. Starting each walk
at the measured pose, not where the last one ended, keeps the commanded velocity the
law's own, so the friction feedforward always has the law's direction, even when the arm
lags. A lead grows only until the spring pushes with MAX_FORCE.

Run from final/src:  sh run.sh run_trials.py <e2t2|sliding|retry> task_<ts>.npz [n_trials]
"""

import multiprocessing
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from arm.control import (
    CONTROL_FREQ_HZ,
    ERGODIC_FREQ_HZ,
    Impedance,
    drain_stdin,
    gravity_compensation,
    keep_rate,
    read_command,
)
from arm.piper import connect_arm, hold_current_pose
from core.agx_pinocchio import AgxPinocchio
from method.law import METHODS, serve
from method.pose import MAX_SPEED, ROTATION_LENGTH, to_pose
from method.task import Task
from robot import OUTPUT_DIR, URDF_PATH, measured_state
from visualization.views import RunView

MAX_FORCE = 5.0  # N, the most the spring may push the key with
TIME_LIMIT = 600.0  # s, a trial's
CYCLES = round(CONTROL_FREQ_HZ / ERGODIC_FREQ_HZ)  # control cycles per law step
PRINT_EVERY = 10  # law steps between progress lines


def place_arm(robot, pin: AgxPinocchio, task: Task, trial: int) -> None:
    """Gravity compensation until Enter, printing the distance to the master's start."""
    drain_stdin()
    cycle = 0
    while True:
        start = time.monotonic()
        q = gravity_compensation(robot, pin)
        if read_command() is not None:
            return
        if cycle % 50 == 0:
            gap = np.linalg.norm(measured_state(pin, task, q) - task.master[0])
            print(
                f"trial {trial + 1}: place the arm at the master's start, {gap:.3f} m away; Enter starts"
            )
        cycle += 1
        keep_rate(start)


def plan(y: np.ndarray, y_next: np.ndarray, lead: float) -> tuple[np.ndarray, np.ndarray]:
    """The commanded state's step per cycle from the measured y towards y_next, and its
    twist (m/s, rad/s).

    Capped at the law's lead, so the command never leads the arm by more than that, even
    when the arm stands outside the cube and y_next is clamped into it. The twist, which
    only the friction feedforward reads, is the law's velocity, at most MAX_SPEED: a lead
    beyond one step is the arm lagging, and it acts through the spring alone.
    """
    dt = 1.0 / ERGODIC_FREQ_HZ
    delta = y_next - y
    length = max(np.linalg.norm(delta), 1e-12)
    velocity = min(1.0, MAX_SPEED * dt / length) * delta / dt
    twist = np.concatenate([velocity[:3], velocity[3:] / ROTATION_LENGTH])
    return min(1.0, lead / length) * delta / CYCLES, twist


class LawProcess:
    """The law in its own process (method.law.serve). A step asked for on one law cycle must
    be back by the next, or the trial raises and main's finally holds the arm."""

    def __init__(self, task_path: Path, method: str, dt: float):
        # spawn, not fork: RunView's thread already runs, and a fork copies its locks held.
        ctx = multiprocessing.get_context("spawn")
        self.conn, child = ctx.Pipe()
        ctx.Process(target=serve, args=(child, task_path, method, dt), daemon=True).start()
        self.conn.recv()  # the law is built
        self.pending = False

    def new_trial(self) -> None:
        if self.pending:  # the last trial ended with a step out; it must not open this one
            self.conn.recv()
        self.conn.send(("trial",))
        self.pending = False

    def ask(self, y: np.ndarray, max_lead: float) -> None:
        if self.pending:
            raise RuntimeError("the law step was not back within one law period")
        self.conn.send(("step", y, max_lead))
        self.pending = True

    def answer(self) -> tuple | None:
        """(y_next, info, seconds) once the step is back, else None; never blocks."""
        if not (self.pending and self.conn.poll()):
            return None
        self.pending = False
        return self.conn.recv()


def take_answer(answer: tuple, sent: tuple, view, trial: int, log: dict) -> tuple:
    """Log and show a step that came back; its walk's step per cycle and twist."""
    y_next, info, step_dt = answer
    t, q, y, tau = sent
    view.update(q, y_next, info.phi, info.w)
    row = {
        "trial": trial,
        "t": t,
        "q": q,
        "y_cmd": y_next,
        "phi": info.phi,
        "stall": info.stall,
        "rho": info.rho,
        "w": info.w,
        "lead": info.lead,
        "tau": tau,
        "step_dt": step_dt,
    }
    for key, value in row.items():
        log[key].append(value)
    if len(log["t"]) % PRINT_EVERY == 1:
        print(
            f"  t {t:5.1f} s  phase {info.phi:.2f}  stall {info.stall:4.1f} s"
            f"  lead {info.lead * 1e3:.1f} mm"
        )
    return plan(y, y_next, info.lead)


def run_trial(robot, setup: tuple, trial: int, log: dict) -> str:
    """One trial from wherever the arm stands: 'success', 'abort' or 'limit'.

    The walk to a law step's state starts on the cycle the step comes back, from the state
    it was asked for, and is CYCLES increments long.
    """
    law, impedance, view, pin, task = setup
    law.new_trial()
    view.new_trial()
    max_lead = MAX_FORCE / impedance.controller.Kc[:3].max()  # m
    t0, cycle, tau = time.monotonic(), 0, np.zeros(6)
    y_cmd = measured_state(pin, task, np.array(robot.get_joint_angles().msg))
    step, twist, left = np.zeros_like(y_cmd), np.zeros(6), 0
    while True:
        start = time.monotonic()
        q = np.array(robot.get_joint_angles().msg)
        if cycle % CYCLES == 0:
            sent = (start - t0, q, measured_state(pin, task, q), tau)
            law.ask(sent[2], max_lead)
        answer = law.answer()
        if answer is not None:
            step, twist = take_answer(answer, sent, view, trial, log)
            y_cmd, left = sent[2], CYCLES
        if left:
            y_cmd, left = y_cmd + step, left - 1
        tau = impedance.send(robot, q, to_pose(y_cmd, task.R0), twist)
        command = read_command()
        if command in ("", "a"):
            return "success" if command == "" else "abort"
        if start - t0 > TIME_LIMIT:
            return "limit"
        # Every torque cycle's own length; a law cycle is one with cycle % CYCLES == 0.
        log["cycle_work"].append((trial, cycle, time.monotonic() - start))
        cycle += 1
        keep_rate(start)


def save(log: dict, outcomes: list[str], method: str, task_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"run_{method}_{datetime.now():%Y%m%d_%H%M%S}.npz"
    np.savez(
        path,
        **{k: np.array(v) for k, v in log.items()},
        outcomes=np.array(outcomes),
        method=method,
        task=str(task_path),
    )
    print(f"saved {len(outcomes)} trials to {path}")
    return path


def prepare(method: str, task_path: Path) -> tuple:
    """Everything slow, built before the arm is connected: the law, the controller, the view."""
    task = Task.load(task_path)
    return (
        LawProcess(task_path, method, 1.0 / ERGODIC_FREQ_HZ),
        Impedance(),
        RunView(task),
        AgxPinocchio(str(URDF_PATH)),
        task,
    )


def main(
    robot, setup: tuple, method: str, task_path: Path, n_trials: int, out_dir: Path = OUTPUT_DIR
) -> Path:
    pin, task = setup[3], setup[4]
    log, outcomes = defaultdict(list), []
    try:
        for trial in range(n_trials):
            place_arm(robot, pin, task, trial)
            print(
                f"trial {trial + 1} of {n_trials}, {method}: Enter on success, a then Enter to abort"
            )
            outcomes.append(run_trial(robot, setup, trial, log))
            print(f"trial {trial + 1}: {outcomes[-1]} after {log['t'][-1]:.1f} s")
    except KeyboardInterrupt:
        print("\ninterrupted, holding the current pose")
        if log["trial"] and len(outcomes) == log["trial"][-1]:
            outcomes.append("interrupted")
    finally:
        hold_current_pose(robot, np.array(robot.get_joint_angles().msg))
    return save(log, outcomes, method, task_path, out_dir)


if __name__ == "__main__":
    method, task_path = sys.argv[1], Path(sys.argv[2])
    if method not in METHODS:
        sys.exit(f"method must be one of {', '.join(METHODS)}")
    setup = prepare(method, task_path)
    main(connect_arm(), setup, method, task_path, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
