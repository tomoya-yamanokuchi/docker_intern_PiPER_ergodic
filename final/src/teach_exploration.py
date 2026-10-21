#!/usr/bin/env python3
"""Teach the exploration distribution: pose the arm, Enter confirms the pose as a datapoint.

TOUCHES THE ARM. Gravity compensation at 100 Hz, so the arm stays backdrivable. One
confirmation is one datapoint; a region is weighted by confirming there again, after
re-posing the arm slightly. "u" withdraws the last datapoint. The MeshCat view shows each
datapoint as a key frame and refits the kernel density on every change.

An existing exploration_*.npz given as the argument is extended; it is only read, and the
session writes a new timestamped file. Ctrl-C holds the arm, then saves.

Run from final/src:  sh run.sh teach_exploration.py [exploration_<timestamp>.npz]
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from arm.control import gravity_compensation, keep_rate, read_command
from arm.piper import connect_arm, hold_current_pose
from core.agx_pinocchio import AgxPinocchio
from robot import OUTPUT_DIR, TCP_FRAME, URDF_PATH
from visualization.views import TeachingView


def apply(command: str, q: np.ndarray, points: list[np.ndarray]) -> bool:
    """Act on one line typed; True if the datapoints changed."""
    if command == "":
        points.append(q)
    elif command == "u" and points:
        points.pop()
    else:
        print(
            "Enter confirms the pose, u withdraws the last datapoint"
            if command != "u"
            else "nothing to withdraw"
        )
        return False
    return True


def main(robot, view: TeachingView, start: Path | None = None, out_dir: Path = OUTPUT_DIR) -> Path:
    pin = AgxPinocchio(str(URDF_PATH))
    points = [] if start is None else list(np.load(start)["q"])
    view.set_datapoints(points)
    print(
        f"{len(points)} datapoints to start from; Enter confirms a pose, u withdraws, Ctrl-C saves"
    )
    try:
        while True:
            start_time = time.monotonic()
            q = gravity_compensation(robot, pin)
            command = read_command()
            if command is not None and apply(command, q, points):
                p, _ = (
                    pin.forward_kinematics(points[-1], TCP_FRAME) if points else (np.zeros(3), None)
                )
                print(f"{len(points)} datapoints; newest {TCP_FRAME} at {np.round(p, 4)} m")
                view.set_datapoints(points)
            view.update(q)
            keep_rate(start_time)
    except KeyboardInterrupt:
        print("\ninterrupted, holding the current pose")
    finally:
        hold_current_pose(robot, np.array(robot.get_joint_angles().msg))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"exploration_{datetime.now():%Y%m%d_%H%M%S}.npz"
    np.savez(path, q=np.array(points).reshape(-1, 6))
    print(f"saved {len(points)} datapoints to {path}")
    return path


if __name__ == "__main__":
    view = TeachingView()  # before connecting, so the first torque is not kept waiting
    main(connect_arm(), view, Path(sys.argv[1]) if len(sys.argv) > 1 else None)
