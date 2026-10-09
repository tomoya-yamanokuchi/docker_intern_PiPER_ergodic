#!/usr/bin/env python3
"""Seed a datapoint set with a tube of poses around a master trajectory.

Hardware-free. The master is a joint_angles_*.npz from record_joint_angles.py. At
stations SPACING apart along it, in the pose metric the phase uses, the set gets the
master's own pose and PER_STATION poses scattered around it: within TUBE_RADIUS in
position and TUBE_TILT in orientation, by closed-form IK seeded with the master's
joint angles. teach_datapoints.py then extends the file, so only the regions that
need extra weight are taught by hand.

The scattered poses were never demonstrated: nothing here knows the fixture, so a
tube wider than the clearance around the master puts target points inside it.

Run from workspace/src:  python seed_datapoints_from_master.py master.npz
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from direct_teaching.distribution.phase_projection import master_phases, tcp_poses
from direct_teaching.recorder.joint_angle_recorder import load_recording, save_datapoints
from kinematics.kinematic_solver import KinematicSolver

URDF_PATH = (
    Path(__file__).resolve().parent / "agx_reference/piper/piper/urdf/piper_description.urdf"
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

SPACING = 0.01  # m of pose distance between stations
# m per rad in that distance, as label_datapoint_phases.py --rotation-length: a
# stage that only turns the peg gets stations too.
ROTATION_LENGTH = 0.06
TUBE_RADIUS = 0.01  # m
TUBE_TILT = 0.1  # rad
PER_STATION = 4  # scattered poses, beside the master's own
# A scattered pose whose joint angles are further than this from the master's is on
# another IK branch, a configuration the arm would not be in there.
BRANCH_TOL = 0.5  # rad


def in_ball(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:  # (n, 3), uniform
    v = rng.normal(size=(n, 3))
    return radius * rng.random((n, 1)) ** (1 / 3) * v / np.linalg.norm(v, axis=1, keepdims=True)


def tube_datapoints(
    solver: KinematicSolver,
    q_master: np.ndarray,  # (N, 6) rad
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:  # (M, 6) rad, and the scattered poses dropped
    p, quat = tcp_poses(solver.pin_model, q_master)
    phi, length = master_phases(p, quat, ROTATION_LENGTH)
    stations = np.searchsorted(phi * length, np.arange(0.0, length, SPACING))
    q, dropped = [], 0
    for n in stations:
        q.append(q_master[n])
        R_n = Rotation.from_quat(quat[n], scalar_first=True)
        dps, rotvecs = in_ball(rng, PER_STATION, TUBE_RADIUS), in_ball(rng, PER_STATION, TUBE_TILT)
        for dp, rotvec in zip(dps, rotvecs, strict=True):
            R = (Rotation.from_rotvec(rotvec) * R_n).as_matrix()
            q_i = solver.inverse_kinematics(p[n] + dp, R, q_init=q_master[n])
            if q_i is None or np.abs(q_i - q_master[n]).max() > BRANCH_TOL:
                dropped += 1
            else:
                q.append(q_i)
    return np.array(q), dropped


def main(master_path: Path) -> None:
    _, q_master = load_recording(master_path)
    q, dropped = tube_datapoints(
        KinematicSolver(str(URDF_PATH)), q_master, np.random.default_rng(0)
    )
    print(
        f"{len(q)} datapoints within {TUBE_RADIUS} m and {TUBE_TILT} rad of the master, "
        f"{dropped} scattered poses dropped (unreachable or on another IK branch)"
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"datapoints_{datetime.now():%Y%m%d_%H%M%S}.npz"
    save_datapoints(path, q)
    print(f"saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("master", type=Path, help="joint_angles_*.npz, the master trajectory")
    main(parser.parse_args().master)
