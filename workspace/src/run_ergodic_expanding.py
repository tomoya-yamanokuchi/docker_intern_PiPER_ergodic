#!/usr/bin/env python3
"""Expanding-target ergodic exploration on the live arm.

run_ergodic_phase.py with the law swapped for ExpandingErgodicController, as in
Ergodic_Exploration_expanding_2D.ipynb: the target is every datapoint up to the
phase reached, a share of it on the newest slice, and the statistic is a time
average that forgets. There is no stall, sigma_b or beta. The task, its cube, the
loop, gains and rates are run_ergodic_phase.py's, unchanged.

The target also holds a sparse band beside the master, CORRIDOR_OFFSET to either
side, so the strip along the demonstrated path is explored where no datapoint was
taught. The datapoints of the entry, those before ENTRY_PHASE, are spread ENTRY_WIDEN
further to either side of the master there, so more approaches to it are tried.

The task is planar: the law explores world x-y only. z and the peg's tilt are
held at the start pose, and rotation about the peg axis is left free.

The phase starts at 0, so place the arm near the master's start pose first, at
the working height and with the peg at the tilt the run should hold.
The run ends when the peg tip comes within GOAL_RADIUS of the master's last position,
in x-y, as a trial of run_ergodic_phase_trials.py does, or on Ctrl-C. Either way the
arm goes to a position hold, and then the run and a phase trace are written.

Run from workspace/src:  python run_ergodic_expanding.py master.npz datapoints_phase.npz
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from controller.feed_forward import FeedForward
from direct_teaching.distribution.phase_projection import corridor_points, widen_across
from direct_teaching.recorder.joint_angle_recorder import load_recording
from ergodic_controller.phase_ergodic_controller import ExpandingErgodicController, PhaseTask
from execution.executor_helpers import connect_arm, hold_current_pose
from execution.live_ergodic_controller import (
    ERGODIC_K,
    TCP_FRAME_NAME,
    U_MAX,
    URDF_PATH,
    Exploration,
    explore,
    first_setpoints,
    prepare_phase_law,
    print_start,
    save_run,
)
from run_ergodic_phase import save_phase_trace
from run_ergodic_phase_trials import GOAL_RADIUS, reached_goal
from visualization.visualizer import LiveView

# The target's share on the newest slice: the notebook's best of 0, 0.25, 0.5 and 0.75.
FRONT_SHARE = 0.25
# The forgetting window, in the steps one sigma_f of progress takes. The notebook's
# best window was 800 steps where sigma_f of progress took 10.7: 75 of them.
FORGET_PROGRESS = 75.0
# The band beside the master: a point on it and one this far to either side, at
# stations this far apart. Each weighs as one datapoint.
CORRIDOR_OFFSET = 0.02  # m
CORRIDOR_SPACING = 0.01  # m
# The search before the slit: the datapoints up to this phase, spread this much further
# to either side at their widest. 0.2 is where 2d_key's master enters the slit.
ENTRY_PHASE = 0.2
ENTRY_WIDEN = 0.01  # m


def with_wider_entry(
    task: PhaseTask,
    X_master: np.ndarray,  # (N, 6) cube states of the master's samples
) -> PhaseTask:
    """The task with its entry datapoints spread further across the master's line there.

    The line is the master's over the 0.05 of phase after ENTRY_PHASE, the way in; its
    samples before that are the demonstration's own approach, which is not straight.
    """
    P = X_master[:, :2] * task.span
    mouth, inside = np.searchsorted(task.phi_master, [ENTRY_PHASE, ENTRY_PHASE + 0.05])
    entry = task.phi < ENTRY_PHASE
    X = task.X.copy()
    wide = widen_across(X[entry, :2] * task.span, P[mouth], P[inside] - P[mouth], ENTRY_WIDEN)
    X[entry, :2] = np.clip(wide / task.span, 0.0, 1.0)
    return replace(task, X=X)


def with_corridor(
    task: PhaseTask,
    X_master: np.ndarray,  # (N, 6) cube states of the master's samples
) -> PhaseTask:
    """The task with the band beside its master added to its datapoints.

    The band is laid out in metres and in x-y; its other cube axes are those of the
    master sample of the same phase.
    """
    points, phi = corridor_points(
        X_master[:, :2] * task.span, task.phi_master, CORRIDOR_SPACING, CORRIDOR_OFFSET
    )
    nearest = np.abs(task.phi_master - phi[:, None]).argmin(axis=1)
    X = X_master[nearest]
    X[:, :2] = np.clip(points / task.span, 0.0, 1.0)
    return replace(task, X=np.vstack([task.X, X]), phi=np.concatenate([task.phi, phi]))


def prepare_expanding_exploration(master: Path, labelled: Path) -> Exploration:
    """prepare_phase_exploration with the expanding law, before the arm is touched."""
    # track_master only for the task's progress_steps: this law's target is the datapoints'.
    controller, distribution, phase_law, master_ends = prepare_phase_law(
        master, labelled, planar=True, track_master=True
    )
    _, q_master = load_recording(master)
    poses = (controller.pin_model.forward_kinematics(q_i, TCP_FRAME_NAME) for q_i in q_master)
    X_master = np.array([distribution.pose_to_state(p_i, R_i) for p_i, R_i in poses])
    task = with_corridor(with_wider_entry(phase_law.task, X_master), X_master)
    forget_window = FORGET_PROGRESS * task.progress_steps
    print(
        f"expanding: front share {FRONT_SHARE:g}, forgetting over {forget_window:.0f} steps, "
        f"{len(task.X) - len(phase_law.task.X)} corridor points beside the master"
    )
    law = ExpandingErgodicController(task, U_MAX, forget_window, FRONT_SHARE, ERGODIC_K)
    return Exploration(
        controller=controller,
        feed_forward=FeedForward(urdf_path=str(URDF_PATH), dofs=6),
        distribution=distribution,
        ergodic=law,
        live=LiveView(
            distribution, URDF_PATH, TCP_FRAME_NAME, phase_law=law, master_ends=master_ends
        ),
        planar=True,
    )


def main(master: Path, labelled: Path) -> None:
    exploration = prepare_expanding_exploration(master, labelled)
    _, q_master = load_recording(master)
    goal, _ = exploration.controller.pin_model.forward_kinematics(q_master[-1], TCP_FRAME_NAME)

    robot = connect_arm()
    joint_angles = np.array(robot.get_joint_angles().msg)
    setpoints = first_setpoints(exploration, joint_angles)
    print_start(exploration, joint_angles, setpoints)
    print(f"exploring until the peg tip is within {GOAL_RADIUS} m of {np.round(goal, 4)} m")
    # (t s, q (6,) rad, p_target (3,) m, R_target (3, 3), tau (6,) N*m) per cycle
    log = []
    try:
        seconds = explore(
            robot, exploration, setpoints, log, stop=lambda q: reached_goal(exploration, q, goal)
        )
        print(f"goal reached after {seconds:.1f} s")
    except KeyboardInterrupt:
        print("\ninterrupted, holding current pose")
    finally:
        # Hold before writing, so a failed write cannot leave the arm unheld.
        hold_current_pose(robot, log[-1][1] if log else joint_angles)
        if log:
            save_run(log, labelled, label="_expanding")
            save_phase_trace(exploration.ergodic.trace)
    print(f"final phase {exploration.ergodic.phi:.3f} after {exploration.ergodic.step_count} steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Both mandatory: the arm explores whatever task these describe, so never pick one implicitly.
    parser.add_argument("master", type=Path, help="joint_angles_*.npz, the master trajectory")
    parser.add_argument(
        "labelled", type=Path, help="datapoints_*_phase.npz from label_datapoint_phases.py"
    )
    args = parser.parse_args()
    main(args.master, args.labelled)
