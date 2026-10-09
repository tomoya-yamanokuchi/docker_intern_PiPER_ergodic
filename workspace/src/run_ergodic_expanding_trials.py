#!/usr/bin/env python3
"""The trialled experiment of run_ergodic_phase_trials.py, run with the expanding law.

Every trial starts the law over: its phase and stall go back to 0, so the target
shrinks back to the entry, and its statistic is emptied, so nothing an earlier trial
covered counts as covered. The trials are independent repeats of one run, and their
times measure the law's spread, not learning over the sequence. Carrying the paths
that ended in a stall into later trials was tried on 2d_key and dropped: they covered
the way in that every trial needs, and the third trial after the first stall never
left the entry.

The goal, the backdriving between trials and the outputs are run_ergodic_phase_trials.py's.
Interactive only, for the reason run_ergodic_trials.py gives.

Run from workspace/src:  python run_ergodic_expanding_trials.py master.npz datapoints_phase.npz 5
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
    print_start,
    save_run,
)
from run_ergodic_expanding import prepare_expanding_exploration
from run_ergodic_phase import save_phase_trace
from run_ergodic_phase_trials import (
    GOAL_RADIUS,
    reached_goal,
    save_trials,
    start_trial,
)
from run_ergodic_trials import backdrive_to_start, report_trials
from visualization.visualizer import cumulative_average


def run_one_trial(
    robot,
    exploration: Exploration,
    labelled: Path,
    q: np.ndarray,  # (6,) rad, the pose this trial starts from
    goal: np.ndarray,  # (3,) m
    trial: int,
) -> float:  # s, from the first torque to the goal
    law = exploration.ergodic
    start_trial(exploration)
    law.memory_x.clear()
    law.memory_phi.clear()
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
        if log:
            save_run(log, labelled, label=f"_expanding_trial{trial}")


def run_trials(
    robot, exploration: Exploration, master: Path, labelled: Path, n_trials: int
) -> None:
    """run_ergodic_phase_trials.run_trials, with this law's trial."""
    path = OUTPUT_DIR / f"ergodic_expanding_trials_{datetime.now():%Y%m%d_%H%M%S}.npz"
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
                f"trial {trial} of {n_trials}: {seconds:.1f} s, cumulative average {average:.1f} s"
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


def main(master: Path, labelled: Path, n_trials: int) -> None:
    exploration = prepare_expanding_exploration(master, labelled)
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
    main(args.master, args.labelled, args.n_trials)
