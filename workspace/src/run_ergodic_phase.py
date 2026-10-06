#!/usr/bin/env python3
"""Phase-augmented ergodic exploration on the live arm.

run_ergodic_pipeline.py with the ergodic law swapped for the phase-conditioned
one of ergodic_controller/phase_ergodic_controller.py, as in
Ergodic_Exploration_phase_key_in_lock_2D.ipynb. A phase read off the master
conditions the target density and the spatial statistic; a stall widens the
backward kernel so earlier-phase targets return. The loop, gains and rates are
execution/live_ergodic_controller.py's, unchanged.

The task is planar: the law explores world x-y only. z and the peg's tilt are
held at the start pose, and rotation about the peg axis is left free, neither
held nor driven. The pose is still recorded and logged in all six DoF.

The phase starts at 0, so place the arm near the master's start pose first, at
the working height and with the peg at the tilt the run should hold.
Ctrl-C hands the arm to a position hold, then writes the run and a phase trace.

Run from workspace/src:  python run_ergodic_phase.py master.npz datapoints_phase.npz
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from execution.executor_helpers import connect_arm, hold_current_pose
from execution.live_ergodic_controller import (
    ERGODIC_FREQ_HZ,
    OUTPUT_DIR,
    explore,
    first_setpoints,
    prepare_phase_exploration,
    print_start,
    save_run,
)

# The only tuning knob of the method: scales T(phi), the stall time sigma_b grows over.
BETA = 1.0


def save_phase_trace(trace: list[tuple[float, float, float]]) -> None:
    """t s, phi, sigma_b and stall / T(phi), one row per ergodic step."""
    phi, sigma_b, stall = (np.array(col) for col in zip(*trace, strict=True))
    path = OUTPUT_DIR / f"phase_trace_{datetime.now():%Y%m%d_%H%M%S}.npz"
    np.savez(
        path, t=np.arange(len(phi)) / ERGODIC_FREQ_HZ, phi=phi, sigma_b=sigma_b, stall_over_T=stall
    )
    print(f"saved {len(phi)} ergodic steps to {path}")


def main(master: Path, labelled: Path) -> None:
    exploration = prepare_phase_exploration(master, labelled, BETA)

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
            save_run(log, labelled, label="_phase")
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
