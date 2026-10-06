#!/usr/bin/env python3
"""Confirm-driven kinesthetic teaching: the operator poses the arm and presses Enter.

The loop is record_joint_angles.py's -- gravity compensation at 100 Hz, so the arm
stays backdrivable -- but nothing is recorded by the clock. A datapoint is a pose
the operator deliberately confirms, which is how the E2T2 paper builds its
reference distribution: importance comes from taking more datapoints in a region,
so pressing Enter repeatedly near the hole is what weights it.

Enter confirms the current pose, "u" withdraws the last one. A region is emphasised
by confirming poses there repeatedly, as the paper gives insertion almost half its
datapoints.

Every change refits the GMM and redraws it in MeshCat with the datapoints on top.
Ctrl-C hands the arm to a position hold, then writes
datapoints_<timestamp>.npz to --out, workspace/output by default.

An existing datapoints file given as the argument is loaded and extended; it is
only read, and the session writes a new timestamped file, so an edit cannot
destroy the set it started from. Withdrawing works from the end of that set.

Run from workspace/src:  python teach_datapoints.py [datapoints_<timestamp>.npz] [--out DIR]
"""

import argparse
import select
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from core.agx_pinocchio import AgxPinocchio
from direct_teaching.recorder.joint_angle_recorder import load_recording, save_datapoints
from execution.executor_helpers import (
    apply_joint_torques,
    connect_arm,
    hold_current_pose,
    read_joint_velocities,
)
from visualization.visualizer import TeachingView

URDF_PATH = (
    Path(__file__).resolve().parent / "agx_reference/piper/piper/urdf/piper_description.urdf"
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

CONTROL_FREQ_HZ = 100.0
TCP_FRAME_NAME = "peg_tcp"
# As run_ergodic_pipeline.py: 8 components, as the E2T2 paper selected.
N_COMPONENTS = 8


def read_command() -> str | None:
    """The line waiting on stdin, stripped, or None if there is none.

    Polled rather than read, because blocking on input() would stop the torque
    stream: the firmware latches the last torque, so the arm would hold a stale
    gravity torque instead of compensating the pose it is being moved into.
    """
    if not select.select([sys.stdin], [], [], 0)[0]:
        return None
    return sys.stdin.readline().strip()


def apply_command(command: str, q: np.ndarray, points: list[np.ndarray]) -> bool:
    """Act on one stdin line; True if the datapoints changed. q (6,) rad."""
    if command == "":
        points.append(np.array(q))
    elif command == "u":
        if not points:
            print("no datapoints to withdraw")
            return False
        points.pop()
    else:
        print(f"unknown command {command!r}; Enter confirms a pose, u withdraws the last")
        return False
    return True


def report(pin: AgxPinocchio, points: list[np.ndarray]) -> None:
    """Count, where the newest datapoint's peg tip sits, and what the fit still needs."""
    if not points:
        print("0 datapoints")
        return
    p, _ = pin.forward_kinematics(points[-1], TCP_FRAME_NAME)
    missing = N_COMPONENTS - len(points)
    fit = f"{missing} more for the first fit" if missing > 0 else "fitted"
    print(f"{len(points)} datapoints, newest {TCP_FRAME_NAME} at {np.round(p, 4)} m; {fit}")


def load_starting_points(
    pin: AgxPinocchio, view: TeachingView, path: Path | None
) -> list[np.ndarray]:
    """The set a session starts from: an existing file's datapoints, or nothing.

    The file is only read. A session always writes a new timestamped one, so an
    edit cannot destroy the set it started from.
    """
    if path is None:
        return []
    points = list(load_recording(path)[1])
    print(f"extending {path}")
    report(pin, points)
    view.set_datapoints(points)
    return points


def run_control_cycle(
    robot,
    pin: AgxPinocchio,
    view: TeachingView,
    points: list[np.ndarray],
    R_world_base: np.ndarray,  # (3, 3)
) -> np.ndarray:  # (6,) rad, the q this cycle measured
    """One gravity-compensation cycle, plus whatever the operator typed into it."""
    joint_angles = np.array(robot.get_joint_angles().msg)
    joint_velocities = read_joint_velocities(robot)

    gravity_torque = pin.inverse_dynamics(
        joint_angles, joint_velocities, np.zeros_like(joint_velocities), R_world_base
    )
    apply_joint_torques(robot, gravity_torque)

    command = read_command()
    if command is not None and apply_command(command, joint_angles, points):
        report(pin, points)
        view.set_datapoints(points)
    view.update(joint_angles)
    return joint_angles


def main(datapoints: Path | None, out_dir: Path) -> None:
    pin = AgxPinocchio(str(URDF_PATH))
    view = TeachingView(URDF_PATH, TCP_FRAME_NAME, N_COMPONENTS)
    points = load_starting_points(pin, view, datapoints)
    robot = connect_arm()
    joint_angles = np.array(robot.get_joint_angles().msg)

    R_world_base = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()
    period = 1.0 / CONTROL_FREQ_HZ

    print(f"q = {np.round(joint_angles, 4)} rad")
    print("gravity compensation on; Enter confirms a pose, u withdraws, Ctrl-C saves")
    try:
        while True:
            start_time = time.monotonic()
            joint_angles = run_control_cycle(robot, pin, view, points, R_world_base)
            elapsed_time = time.monotonic() - start_time
            if elapsed_time < period:
                time.sleep(period - elapsed_time)
            else:
                print(f"warning: control loop overrun {elapsed_time:.3f}s > {period:.3f}s")

    except KeyboardInterrupt:
        print("\ninterrupted, holding current pose")
    finally:
        # Hold before any file I/O, so a failed write cannot leave the arm unheld.
        hold_current_pose(robot, joint_angles)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"datapoints_{datetime.now():%Y%m%d_%H%M%S}.npz"
        save_datapoints(path, points)
        print(f"saved {len(points)} datapoints to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datapoints",
        nargs="?",
        type=Path,
        help="datapoints_*.npz to extend; omit to start a new set. Only ever read.",
    )
    parser.add_argument(
        "--out", type=Path, default=OUTPUT_DIR, help="directory to save in, created if missing"
    )
    args = parser.parse_args()
    main(args.datapoints, args.out)
