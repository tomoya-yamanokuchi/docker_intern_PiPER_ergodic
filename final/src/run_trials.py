#!/usr/bin/env python3
"""Run the method on the arm: N trials of the locked-door task, one method throughout.

TOUCHES THE ARM. Methods: e2t2 (static target, its statistic handed on across trials, as
its paper does), expanding (expanding target with E2T2's statistic), sliding (sliding
target with forgetting, fresh each trial) and retry (sliding with the retry). Every trial starts at the master's start, held there until Enter
(or for SETTLE s with --auto). After every trial the reset gets it there: the master played
back to its start from the sample nearest the arm, shifted by the arm's offset from that
sample, closing the door and taking the key out; then a walk removes the offset. Only a
run that starts away from the master's start, or a reset that fails, needs the operator:
the arm is backdrivable; close the door, and either press Enter with the key still in the
lock, or take it out and put the arm near the master's start (the distance is printed)
first. Either way it walks the rest along the master. A run that starts
where the last one's reset ended needs no one, so sequences can follow each other:

  for i in 1 2 3; do for m in e2t2 retry; do sh run.sh run_trials.py $m <task> 4 --auto || break 2; done; done

The law is not stepped meanwhile, so neither the reset nor the handling enters the
statistic.

A trial is a success once the phase passes SUCCESS_PHASE, and a failure at TIME_LIMIT;
Enter ends it as an abort, for a trial the operator had to touch. Ctrl-C holds the arm and
saves the trials so far to run_<method>_<timestamp>.npz in final/output/evaluation; view one with
view_run.py.

Control is fully online, as closed-loop ergodic control is meant to be: every
ERGODIC_FREQ_HZ the measured pose goes into the law and its next state comes out,
MAX_SPEED / ERGODIC_FREQ_HZ away plus whatever the arm left of the last step, and the
commanded pose walks from the measured pose to it at CONTROL_FREQ_HZ. Starting each walk
at the measured pose, not where the last one ended, keeps the commanded velocity the
law's own, so the friction feedforward always has the law's direction, even when the arm
lags. A lead grows only until the spring pushes with MAX_FORCE.

Run from final/src:  sh run.sh run_trials.py <e2t2|sliding|retry> task_<ts>.npz [n_trials] [--auto]
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
from method.law import METHODS, TOLERANCE, serve
from method.pose import MAX_SPEED, ROTATION_LENGTH, to_pose
from method.task import Task
from robot import OUTPUT_DIR, URDF_PATH, measured_state
from visualization.views import RunView

MAX_FORCE = 5.0  # N, the most the spring may push the key with
# N, the same during a reset: the key sticks in the keyhole on the way out, and a reset is
# rig, not method, so it may pull harder than a trial pushes.
RESET_FORCE = 15.0
TIME_LIMIT = 600.0  # s, a trial's
SUCCESS_PHASE = 0.9
RESET_LIMIT = 60.0  # s, a walk's; the whole master at full speed takes about 16 s
SETTLE = 2.0  # s, the hold at the master's start before an automatic trial
AUTO = False  # --auto: start every trial after SETTLE, not on Enter
# Every arm run lands here, so the experiment is one glob; simulate.py writes to its own.
EVALUATION_DIR = OUTPUT_DIR / "evaluation"
CYCLES = round(CONTROL_FREQ_HZ / ERGODIC_FREQ_HZ)  # control cycles per law step
PRINT_EVERY = 10  # law steps between progress lines


def place_arm(robot, pin: AgxPinocchio, task: Task, trial: int) -> np.ndarray:
    """Gravity compensation until Enter, printing the distance to the master's start; (d,),
    the arm's state at Enter."""
    drain_stdin()
    cycle = 0
    while True:
        start = time.monotonic()
        q = gravity_compensation(robot, pin)
        if read_command() is not None:
            return measured_state(pin, task, q)
        if cycle % 50 == 0:
            gap = np.linalg.norm(measured_state(pin, task, q) - task.master[0])
            print(
                f"trial {trial + 1}: place the arm at the master's start, {gap:.3f} m away;"
                " Enter walks it there"
            )
        cycle += 1
        keep_rate(start)


def walk_path(y: np.ndarray, via: np.ndarray) -> np.ndarray:
    """(n, d), from y through the states via (m, d), one control cycle at MAX_SPEED apart."""
    corners = np.vstack([y, via])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(corners, axis=0), axis=1))])
    grid = np.linspace(0.0, s[-1], int(np.ceil(s[-1] * CONTROL_FREQ_HZ / MAX_SPEED)) + 1)
    return np.column_stack([np.interp(grid, s, c) for c in corners.T])


def measure(robot, pin: AgxPinocchio, task: Task) -> np.ndarray:  # (d,) m, the arm's state now
    return measured_state(pin, task, np.array(robot.get_joint_angles().msg))


def reset_via(task: Task, y: np.ndarray, success: bool) -> np.ndarray:
    """(m, d): from the measured y, the master back to its start, kept at y's offset from its
    nearest sample so a key that ended beside the keyhole withdraws beside it; then the start.

    After a success the key is in the lock, and the offset is only the push's lag: kept, it
    tilts the key in the keyhole and jams it. So it fades within one resolvable width.
    """
    j = int(np.linalg.norm(task.master - y, axis=1).argmin())
    fade = np.ones(j + 1)
    if success:
        fade = np.clip(1 - np.arange(j + 1) / (task.sigma_f * (len(task.master) - 1)), 0, 1)
    back = task.master[j::-1] + np.outer(fade, y - task.master[j])
    return np.vstack([back, task.master[:1]])


def walk(robot, setup: tuple, via: np.ndarray, trial: int, log: dict) -> bool:
    """Walk the commanded pose from the measured one through via; True once the arm is
    within TOLERANCE of via's end, False at RESET_LIMIT.

    The commanded pose waits while the arm lags it by more than the lead at which the
    spring pulls with RESET_FORCE. The law is not stepped.
    """
    _, impedance, _, pin, task = setup
    max_lead = RESET_FORCE / impedance.controller.Kc[:3].max()  # m
    path = walk_path(measure(robot, pin, task), via)
    t0, i, done = time.monotonic(), 0, False
    while not done and (t := time.monotonic() - t0) < RESET_LIMIT:
        start = time.monotonic()
        q = np.array(robot.get_joint_angles().msg)
        y = measured_state(pin, task, q)
        if i + 1 < len(path) and np.linalg.norm(path[i + 1] - y) <= max_lead:
            i += 1
        # As in a trial, the twist points from the arm to where the command goes even while
        # the arm stands: the friction feedforward reads only it, and a standing arm a few
        # mm short is held there by friction the spring alone cannot beat.
        twist = plan(y, path[min(i + 1, len(path) - 1)], max_lead)[1]
        impedance.send(robot, q, to_pose(path[i], task.R0), twist)
        for key, value in (("reset_trial", trial), ("reset_t", t), ("reset_q", q)):
            log[key].append(value)
        gap = np.linalg.norm(y - via[-1])
        done = i + 1 == len(path) and gap < TOLERANCE
        if len(log["reset_t"]) % CONTROL_FREQ_HZ == 0:
            print(f"  walk: t {t:4.1f} s, {gap * 1e3:.1f} mm from its end")
        keep_rate(start)
    seconds = time.monotonic() - t0
    log["reset_done"].append((trial, done, seconds))
    print(f"  walk {'done' if done else 'FAILED'} after {seconds:.1f} s")
    return done


def hold_start(robot, setup: tuple) -> None:
    """Hold the master's start under impedance until Enter, or for SETTLE s if AUTO."""
    impedance, task = setup[1], setup[4]
    if not AUTO:
        drain_stdin()
        print("at the master's start; Enter starts the trial")
    t0 = time.monotonic()
    while not (time.monotonic() - t0 > SETTLE if AUTO else read_command() is not None):
        start = time.monotonic()
        q = np.array(robot.get_joint_angles().msg)
        impedance.send(robot, q, to_pose(task.master[0], task.R0), np.zeros(6))
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
    """The law in its own process (method.law.serve). A step not back by the next law cycle
    is late, and that cycle asks for none."""

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
    view.update(q, info.phi, info.w)
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


def ended(t: float) -> str | None:
    """'abort' on the operator's Enter, 'limit' t s into a trial past TIME_LIMIT, else None."""
    if read_command() is not None:
        return "abort"
    return "limit" if t > TIME_LIMIT else None


def run_trial(robot, setup: tuple, trial: int, log: dict) -> str:
    """One trial from wherever the arm stands: 'success', 'abort' or 'limit'.

    The walk to a law step's state starts on the cycle the step comes back, from the state
    it was asked for, and is CYCLES increments long.
    """
    law, impedance, view, pin, task = setup
    law.new_trial()
    max_lead = MAX_FORCE / impedance.controller.Kc[:3].max()  # m
    t0, cycle, tau = time.monotonic(), 0, np.zeros(6)
    y_cmd = measured_state(pin, task, np.array(robot.get_joint_angles().msg))
    step, twist, left = np.zeros_like(y_cmd), np.zeros(6), 0
    while True:
        start = time.monotonic()
        q = np.array(robot.get_joint_angles().msg)
        if cycle % CYCLES == 0:
            # A late step costs the law one period, not the trial: the walk holds meanwhile.
            if law.pending:
                log["late"].append((trial, cycle))
            else:
                sent = (start - t0, q, measured_state(pin, task, q), tau)
                law.ask(sent[2], max_lead)
        answer = law.answer()
        if answer is not None:
            step, twist = take_answer(answer, sent, view, trial, log)
            y_cmd, left = sent[2], CYCLES
            if answer[1].phi > SUCCESS_PHASE:
                return "success"
        if left:
            y_cmd, left = y_cmd + step, left - 1
        tau = impedance.send(robot, q, to_pose(y_cmd, task.R0), twist)
        if (end := ended(start - t0)) is not None:
            return end
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
    robot, setup: tuple, method: str, task_path: Path, n_trials: int, out_dir: Path = EVALUATION_DIR
) -> Path:
    pin, task = setup[3], setup[4]
    log, outcomes, interrupted = defaultdict(list), [], False
    # The last run's reset left the arm here: no one needs to place it.
    ready = np.linalg.norm(measure(robot, pin, task) - task.master[0]) < TOLERANCE
    try:
        for trial in range(n_trials):
            while not ready:
                # Along the master, not straight: the key may still be in the lock.
                via = reset_via(task, place_arm(robot, pin, task, trial), success=True)
                ready = walk(robot, setup, via, trial, log)
            hold_start(robot, setup)
            print(f"trial {trial + 1} of {n_trials}, {method}: Enter aborts")
            outcomes.append(run_trial(robot, setup, trial, log))
            print(f"trial {trial + 1}: {outcomes[-1]} after {log['t'][-1]:.1f} s")
            print("  reset: the master back to its start")
            via = reset_via(task, measure(robot, pin, task), outcomes[-1] == "success")
            ready = walk(robot, setup, via, trial, log)
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted, holding the current pose")
        if log["trial"] and len(outcomes) == log["trial"][-1]:
            outcomes.append("interrupted")
    except Exception:
        if log["trial"] and len(outcomes) == log["trial"][-1]:
            outcomes.append("error")
        raise
    finally:
        hold_current_pose(robot, np.array(robot.get_joint_angles().msg))
        # Here, not after: a trial that raises must not lose the sequence's earlier trials.
        path = save(log, outcomes, method, task_path, out_dir)
    if interrupted:
        sys.exit(130)  # 128 + SIGINT, as an uncaught Ctrl-C exits: a shell loop's || break stops
    return path


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--auto"]
    method, task_path = args[0], Path(args[1])
    if method not in METHODS:
        sys.exit(f"method must be one of {', '.join(METHODS)}")
    setup = prepare(method, task_path)
    AUTO = "--auto" in sys.argv
    main(connect_arm(), setup, method, task_path, int(args[2]) if len(args) > 2 else 1)
