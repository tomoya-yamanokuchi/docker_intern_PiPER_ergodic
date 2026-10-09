#!/usr/bin/env python3
"""Expanding-target ergodic exploration in all six DoF.

run_ergodic_expanding.py with the law on all six cube axes, position and orientation,
rotation about the peg included, and the phase projected on the full pose: the
product metric |dp|^2 + (l * theta)^2 with theta the geodesic rotation angle, l read
from the labelled file. Label with a nonzero l, e.g.

    python label_datapoint_phases.py master.npz datapoints.npz --rotation-length 0.06

With l = 0 a stage that only turns the peg earns no phase, and the target never
grows past it.

The target is the datapoints as taught. The planar run's band beside the master and
its wider entry are laid out in the x-y plane and are not applied; spread at the
entry, in position and in tilt, has to be in the datapoints.

The run ends when the peg tip comes within GOAL_RADIUS of the master's last position
in xyz, orientation not checked, or on Ctrl-C. The phase starts at 0, so place the arm
near the master's start pose first.

Run from workspace/src:  python run_ergodic_expansion_6dof.py master.npz datapoints_phase.npz
"""

import argparse
from pathlib import Path

from run_ergodic_expanding import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Both mandatory: the arm explores whatever task these describe, so never pick one implicitly.
    parser.add_argument("master", type=Path, help="joint_angles_*.npz, the master trajectory")
    parser.add_argument(
        "labelled", type=Path, help="datapoints_*_phase.npz from label_datapoint_phases.py"
    )
    args = parser.parse_args()
    main(args.master, args.labelled, planar=False)
