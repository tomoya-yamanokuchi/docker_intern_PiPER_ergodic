"""The phase-conditioned datapoint distribution over phi, with a slider, in MeshCat."""

import argparse
from pathlib import Path

from visualization.visualizer import Visualizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("master", type=Path, help="joint_angles_*.npz, the master trajectory")
parser.add_argument(
    "labelled", type=Path, help="datapoints_*_phase.npz from label_datapoint_phases.py"
)
args = parser.parse_args()
# Held: the MeshCat server dies with its Visualizer object.
viewer = Visualizer().show_phase_distribution(args.master, args.labelled)
input("Enter to close the viewer: ")
