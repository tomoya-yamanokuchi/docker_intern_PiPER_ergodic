#!/usr/bin/env python3
"""The trialled experiment of run_ergodic_trials.py, run with the phase-conditioned law.

Does the phase law get faster over trials because it remembers what it has tried?
One PhaseErgodicController lives for the whole sequence. A trial boundary resets
its phase clock -- phi, the stall counter, the progress mark and the lag -- because the arm
starts the task over, but never its memory of past states. The spatial statistic
at phase phi weighs every past state by how close its phase is to phi, so in trial
k it already holds what trials 1 .. k-1 tried at that phase. A dead end explored
at phase 0.3 in trial 1 counts as covered when trial 2 reaches phase 0.3.

Only what a trial tried while stuck is carried into the next one: the states it
recorded once a stall had outlasted T(phi). Carrying every state made later trials
slower in the first sequences on the arm, because the route that worked counted as
covered too and the law steered off it -- in peg_in_hole, off the hole's axis.

The goal is the master's last peg_tcp position: a trial ends when the peg tip comes
within GOAL_RADIUS of it -- in x-y for the planar task, whose z is held at the start
pose, and in xyz for the 6-DoF one. Position only, as in run_ergodic_trials.py.

Between trials the arm goes backdrivable; place it back at the pose trial 1 started
from and press Enter. The law is not stepped meanwhile, so the operator's handling
path never enters its memory. Ctrl-C holds the arm and keeps the finished trials.

Interactive only, for the reason run_ergodic_trials.py gives.

This script runs the planar law; run_ergodic_phase_6dof_trials.py runs the 6-DoF one.

Run from workspace/src:  python run_ergodic_phase_trials.py master.npz datapoints_phase.npz 5
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from direct_teaching.recorder.joint_angle_recorder import load_recording
from execution.executor_helpers import connect_arm, hold_current_pose
from execution.live_ergodic_controller import (
    OUTPUT_DIR,
    TCP_FRAME_NAME,
    Exploration,
    explore,
    first_setpoints,
    prepare_phase_exploration,
    print_start,
    save_run,
)
from run_ergodic_phase import BETA, TRACK_MASTER, save_phase_trace
from run_ergodic_trials import backdrive_to_start, report_trials
from visualization.visualizer import cumulative_average

# The two_gaps_obstructed runs that reached the end stopped about 8.5 mm from the
# master's last position, so this is about the closest the law gets unprompted.
GOAL_RADIUS = 0.01  # m
# A state is one the trial tried while stuck once stall / T(phi) has reached this: the
# stall has then lasted as long as the datapoints' share of that phase.
STUCK_STALL = 1.0


def start_trial(exploration: Exploration) -> None:
    """The phase clock back to the task's start; memory_x and memory_phi are kept."""
    law = exploration.ergodic
    law.phi, law.stall, law.last_progress, law.lag = 0.0, 0, 0.0, 0.0


def forget_progress(exploration: Exploration, n_before: int) -> None:
    """Drop this trial's states recorded while it was not stuck; n_before older ones stay.

    The law appends one trace row per state, so the trace's tail is this trial's.
    """
    law = exploration.ergodic
    n_trial = len(law.memory_x) - n_before
    stuck = [stall_over_T >= STUCK_STALL for _, _, stall_over_T in law.trace[-n_trial:]]
    for memory in (law.memory_x, law.memory_phi):
        memory[n_before:] = [m for m, keep in zip(memory[n_before:], stuck, strict=True) if keep]


def reached_goal(exploration: Exploration, q: np.ndarray, goal: np.ndarray) -> bool:
    """q (6,) rad, goal (3,) m: the peg tip within GOAL_RADIUS, on the axes the law moves."""
    p, _ = exploration.controller.pin_model.forward_kinematics(q, TCP_FRAME_NAME)
    axes = 2 if exploration.planar else 3
    return bool(np.linalg.norm(p[:axes] - goal[:axes]) <= GOAL_RADIUS)


def run_one_trial(
    robot,
    exploration: Exploration,
    labelled: Path,
    q: np.ndarray,  # (6,) rad, the pose this trial starts from
    goal: np.ndarray,  # (3,) m
    trial: int,
) -> float:  # s, from the first torque to the goal
    start_trial(exploration)
    n_before = len(exploration.ergodic.memory_x)
    setpoints = first_setpoints(exploration, q)
    print_start(exploration, q, setpoints)
    print(
        f"trial {trial}: exploring until the peg tip is within {GOAL_RADIUS} m of {np.round(goal, 4)} m"
    )
    log = []
    try:
        return explore(
            robot, exploration, setpoints, log, stop=lambda q: reached_goal(exploration, q, goal)
        )
    finally:
        forget_progress(exploration, n_before)
        if log:
            save_run(log, labelled, label=f"_phase_trial{trial}")


def save_trials(path: Path, master: Path, labelled: Path, trials: list[tuple]) -> None:
    """Times, start poses, and the ergodic step each trial ended on, to cut the phase trace."""
    seconds, q_start, end_steps = zip(*trials, strict=True)
    np.savez(
        path,
        master=str(master),
        recording=str(labelled),
        trial_times=np.array(seconds),
        q_start=np.array(q_start),
        trial_end_steps=np.array(end_steps),
    )
    print(f"saved {len(seconds)} trials to {path}")


def run_trials(
    robot, exploration: Exploration, master: Path, labelled: Path, n_trials: int
) -> None:
    """run_ergodic_trials.run_trials, with the phase law's goal and outputs."""
    path = OUTPUT_DIR / f"ergodic_phase_trials_{datetime.now():%Y%m%d_%H%M%S}.npz"
    _, q_master = load_recording(master)
    goal, _ = exploration.controller.pin_model.forward_kinematics(q_master[-1], TCP_FRAME_NAME)
    # (s to the goal, start q (6,) rad, ergodic step count at the end) per finished trial
    trials: list[tuple] = []
    q = q_start = np.array(robot.get_joint_angles().msg)
    try:
        for trial in range(1, n_trials + 1):
            if trial > 1:
                q = backdrive_to_start(robot, exploration, q_start, trial)
            seconds = run_one_trial(robot, exploration, labelled, q, goal, trial)
            trials.append((seconds, q, exploration.ergodic.step_count))
            average = cumulative_average([s for s, *_ in trials])[-1]
            print(
                f"trial {trial} of {n_trials}: {seconds:.1f} s, cumulative average {average:.1f} s, "
                f"{len(exploration.ergodic.memory_x)} states remembered"
            )
    except KeyboardInterrupt:
        print("\ninterrupted, holding current pose")
    finally:
        # As run_ergodic_trials.run_trials: read the arm, never reuse a stale q.
        measured = robot.get_joint_angles()
        hold_current_pose(robot, np.array(measured.msg) if measured is not None else q)
        if trials:
            save_trials(path, master, labelled, trials)
        save_phase_trace(exploration.ergodic.trace)
    report_trials(path, trials)


def main(master: Path, labelled: Path, n_trials: int, planar: bool) -> None:
    exploration = prepare_phase_exploration(master, labelled, BETA, planar, TRACK_MASTER)
    robot = connect_arm()
    run_trials(robot, exploration, master, labelled, n_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Mandatory: the arm explores whatever task these describe, so never pick one implicitly.
    parser.add_argument("master", type=Path, help="joint_angles_*.npz, the master trajectory")
    parser.add_argument(
        "labelled", type=Path, help="datapoints_*_phase.npz from label_datapoint_phases.py"
    )
    parser.add_argument("n_trials", type=int, help="attempts to run")
    args = parser.parse_args()
    main(args.master, args.labelled, args.n_trials, planar=True)
