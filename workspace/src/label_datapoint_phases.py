#!/usr/bin/env python3
"""Label a taught datapoint set with its phase on a master trajectory.

Hardware-free. The master is a joint_angles_*.npz from record_joint_angles.py,
one continuous demonstration of the whole task; each datapoint gets the phase
of its nearest master sample (direct_teaching/distribution/phase_projection.py).
The datapoint file is only read; the labels go to <its name>_phase.npz beside it.

Run from workspace/src:  python label_datapoint_phases.py master.npz datapoints.npz
"""

import argparse
from pathlib import Path

import numpy as np

from core.agx_pinocchio import AgxPinocchio
from direct_teaching.distribution.phase_projection import phase_context, project, save_phase_labels
from direct_teaching.recorder.joint_angle_recorder import load_recording

URDF_PATH = (
    Path(__file__).resolve().parent / "agx_reference/piper/piper/urdf/piper_description.urdf"
)


def main(master_path: Path, datapoints_path: Path) -> None:
    _, q = load_recording(datapoints_path)
    context = phase_context(master_path, q, AgxPinocchio(str(URDF_PATH)))
    phi = project(context.S, context.S_master, context.phi_master, context.scale)

    print(f"master: {len(context.phi_master)} samples; datapoints: {len(q)}")
    print(f"per-axis std [m m m rad rad rad]: {np.round(context.scale, 4)}")
    print(f"k = {context.k}, h = {context.h:.3f}, dl/dphi = {context.dl_dphi:.1f}")
    print(f"sigma_f = lead = {context.sigma_f:.4f}")
    counts, _ = np.histogram(phi, bins=10, range=(0.0, 1.0))
    print(f"datapoints per phi decile: {counts.tolist()}")

    path = datapoints_path.with_name(f"{datapoints_path.stem}_phase.npz")
    save_phase_labels(path, q, phi)
    print(f"saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("master", type=Path, help="joint_angles_*.npz, the master trajectory")
    parser.add_argument("datapoints", type=Path, help="datapoints_*.npz to label")
    args = parser.parse_args()
    main(args.master, args.datapoints)
