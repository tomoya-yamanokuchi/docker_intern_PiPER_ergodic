#!/usr/bin/env python3
"""run_ergodic_phase_trials.py with the 6-DoF law of run_ergodic_phase_6dof.py.

N trials, one memory of past states across them, each trial ending when the peg tip
is within GOAL_RADIUS of the master's last position in xyz. Label with a nonzero l
(label_datapoint_phases.py --rotation-length), as for run_ergodic_phase_6dof.py.

Run from workspace/src:  python run_ergodic_phase_6dof_trials.py master.npz datapoints_phase.npz 5
"""

import argparse
from pathlib import Path

from run_ergodic_phase_trials import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Mandatory: the arm explores whatever task these describe, so never pick one implicitly.
    parser.add_argument("master", type=Path, help="joint_angles_*.npz, the master trajectory")
    parser.add_argument(
        "labelled", type=Path, help="datapoints_*_phase.npz from label_datapoint_phases.py"
    )
    parser.add_argument("n_trials", type=int, help="attempts to run")
    args = parser.parse_args()
    main(args.master, args.labelled, args.n_trials, planar=False)
