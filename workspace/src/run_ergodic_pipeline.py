#!/usr/bin/env python3
"""Ergodic exploration of a recording's pose distribution on the live arm.

Fits the time-weighted PoseDistribution to a recording, computes the E2T2
coefficients, and lets the ergodic controller move a Cartesian target that the
main_tast_imp.py impedance law tracks. This is one open-ended run; the E2T2
paper's trialled experiment is run_ergodic_trials.py.

The loop, the gains and the two rates live in execution/live_ergodic_controller.py,
whose docstring carries the design rationale. This file is only the entry point.

Ctrl-C hands the arm to a position hold.

Run from workspace/src:  python run_ergodic_pipeline.py recording.npz
"""

import argparse
from pathlib import Path

import numpy as np

from execution.executor_helpers import connect_arm, hold_current_pose
from execution.live_ergodic_controller import (
    explore,
    first_setpoints,
    prepare_exploration,
    print_start,
    save_run,
)


def main(recording: Path) -> None:
    exploration = prepare_exploration(recording)

    robot = connect_arm()
    joint_angles = np.array(robot.get_joint_angles().msg)
    setpoints = first_setpoints(exploration, joint_angles)
    print_start(exploration, joint_angles, setpoints)
    print("exploring; Ctrl-C to stop")
    # (t s, q (6,) rad, p_target (3,) m, R_target (3, 3), tau (6,) N*m) per cycle
    log = []
    try:
        explore(robot, exploration, setpoints, log)
    except KeyboardInterrupt:
        print("\ninterrupted, holding current pose")
    finally:
        # Hold before writing, so a failed write cannot leave the arm unheld.
        hold_current_pose(robot, log[-1][1] if log else joint_angles)
        if log:
            save_run(log, recording)
    if exploration.ergodic.step_count:
        print(f"ergodic metric: {exploration.ergodic.ergodic_metric():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Mandatory: the arm explores whatever recording this is, so never pick one implicitly.
    parser.add_argument("recording", type=Path, help="joint_angles_*.npz")
    args = parser.parse_args()
    main(args.recording)
