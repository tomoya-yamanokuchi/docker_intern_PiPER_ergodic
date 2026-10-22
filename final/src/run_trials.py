#!/usr/bin/env python3
"""Run the method on the arm: N trials of the locked-door task, one method throughout.

TOUCHES THE ARM. Methods: e2t2 (static target, its statistic handed on across trials, as
its paper does), sliding (sliding target with forgetting, fresh each trial) and retry
(sliding with the retry) and recall (sliding whose target keeps the whole past, method/recall.py). Before every trial the arm is backdrivable: re-lock the door,
take the key out, put the arm near the master's start (the distance is printed) and press
Enter. The law is not stepped meanwhile, so the handling never enters the statistic.

During a trial, Enter ends it as a success and "a" as a failure; so does the time limit,
20 passes along the master at full speed. Ctrl-C holds the arm and saves the trials so far
to run_<method>_<timestamp>.npz in final/output; view one with view_run.py.

Control is fully online, as closed-loop ergodic control is meant to be: every
ERGODIC_FREQ_HZ the measured pose goes into the law and its next state comes out,
MAX_SPEED / ERGODIC_FREQ_HZ away plus whatever the arm left of the last step, and the
commanded pose walks from the measured pose to it at CONTROL_FREQ_HZ. Starting each walk
at the measured pose, not where the last one ended, keeps the commanded velocity the
law's own, so the friction feedforward always has the law's direction, even when the arm
lags. A lead grows only until the spring pushes with MAX_FORCE.

Run from final/src:  sh run.sh run_trials.py <e2t2|sliding|retry|recall> task_<ts>.npz [n_trials]
"""

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
from method.law import METHODS, Law
from method.pose import MAX_SPEED, ROTATION_LENGTH, to_metric, to_pose
from method.recall import RecallLaw
from method.task import Task
from robot import OUTPUT_DIR, TCP_FRAME, URDF_PATH
from visualization.views import RunView

MAX_FORCE = 5.0  # N, the most the spring may push the key with
PASSES = 20  # a trial's time limit, in passes along the master at full speed
CYCLES = round(CONTROL_FREQ_HZ / ERGODIC_FREQ_HZ)  # control cycles per law step
PRINT_EVERY = 10  # law steps between progress lines


def measured_state(pin: AgxPinocchio, task: Task, q: np.ndarray) -> np.ndarray:  # (6,) m
    p, rot = pin.forward_kinematics(q, TCP_FRAME)
    return to_metric(p[None], rot[None], task.R0)[0]


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


def run_trial(robot, setup: tuple, trial: int, log: dict) -> str:
    """One trial from wherever the arm stands: 'success', 'abort' or 'limit'."""
    law, impedance, view, pin, task = setup
    law.new_trial()
    view.new_trial()
    time_limit = PASSES * task.S_m / MAX_SPEED
    max_lead = MAX_FORCE / impedance.controller.Kc[:3].max()  # m
    t0, cycle, tau = time.monotonic(), 0, np.zeros(6)
    while True:
        start = time.monotonic()
        q = np.array(robot.get_joint_angles().msg)
        if cycle % CYCLES == 0:
            y_cmd = measured_state(pin, task, q)
            y_next, info = law.step(y_cmd, max_lead)
            step, twist = plan(y_cmd, y_next, info.lead)
            view.update(q, y_next, info.phi, info.w)
            for key, value in zip(
                ("trial", "t", "q", "y_cmd", "phi", "stall", "rho", "w", "lead", "tau"),
                (
                    trial,
                    start - t0,
                    q,
                    y_next,
                    info.phi,
                    info.stall,
                    info.rho,
                    info.w,
                    info.lead,
                    tau,
                ),
                strict=True,
            ):
                log[key].append(value)
            if cycle % (CYCLES * PRINT_EVERY) == 0:
                print(
                    f"  t {start - t0:5.1f} s  phase {info.phi:.2f}  stall {info.stall:4.1f} s"
                    f"  lead {info.lead * 1e3:.1f} mm"
                )
        y_cmd = y_cmd + step
        tau = impedance.send(robot, q, to_pose(y_cmd, task.R0), twist)
        command = read_command()
        if command in ("", "a"):
            return "success" if command == "" else "abort"
        if start - t0 > time_limit:
            return "limit"
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
        (RecallLaw if method == "recall" else Law)(task, method, 1.0 / ERGODIC_FREQ_HZ),
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
