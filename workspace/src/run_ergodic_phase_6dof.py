#!/usr/bin/env python3
"""Phase-augmented ergodic exploration in all six DoF, for peg-in-hole.

run_ergodic_phase.py with the law on all six cube axes, position and orientation,
rotation about the peg included, and the phase projected on the full pose: the
product metric |dp|^2 + (l * theta)^2 with theta the geodesic rotation angle, l
read from the labelled file. Label with a nonzero l, e.g.

    python label_datapoint_phases.py master.npz datapoints.npz --rotation-length 0.06

A jammed peg retries without a rule of its own: the stall freezes phi, sigma_b
widens, and the target takes in earlier-phase datapoints -- the top of the hole --
so the law backs the peg out and approaches again.

The phase starts at 0, so place the arm near the master's start pose first.
Ctrl-C hands the arm to a position hold, then writes the run and a phase trace.

Run from workspace/src:  python run_ergodic_phase_6dof.py master.npz datapoints_phase.npz
"""

import argparse
from pathlib import Path

from run_ergodic_phase import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Both mandatory: the arm explores whatever task these describe, so never pick one implicitly.
    parser.add_argument("master", type=Path, help="joint_angles_*.npz, the master trajectory")
    parser.add_argument(
        "labelled", type=Path, help="datapoints_*_phase.npz from label_datapoint_phases.py"
    )
    args = parser.parse_args()
    main(args.master, args.labelled, planar=False)
