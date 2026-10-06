#!/usr/bin/env python3
"""Record joint angles while the arm is backdriven under full gravity compensation.

The loop is agx_reference/piper/main_gc.py; the only addition is a 10 Hz tap on
the q it already reads every cycle. Ctrl-C hands the arm to a position hold,
then writes joint_angles_<timestamp>.npz to --out, workspace/output by default.

Run from workspace/src:  python record_joint_angles.py [--out DIR]
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from core.agx_pinocchio import AgxPinocchio
from direct_teaching.recorder.joint_angle_recorder import JointAngleRecorder
from execution.executor_helpers import (
    apply_joint_torques,
    connect_arm,
    hold_current_pose,
    read_joint_velocities,
)

URDF_PATH = (
    Path(__file__).resolve().parent / "agx_reference/piper/piper/urdf/piper_description.urdf"
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

REC_FREQ_HZ = 100.0
CONTROL_FREQ_HZ = 100.0


def main(out_dir: Path) -> None:
    pin = AgxPinocchio(str(URDF_PATH))
    robot = connect_arm()
    joint_angles = np.array(robot.get_joint_angles().msg)

    R_world_base = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()
    recorder = JointAngleRecorder(REC_FREQ_HZ)
    period = 1.0 / CONTROL_FREQ_HZ

    print(f"q = {np.round(joint_angles, 4)} rad")
    print(f"gravity compensation on, recording at {REC_FREQ_HZ:g} Hz; Ctrl-C to stop and save")
    try:
        while True:
            start_time = time.monotonic()

            joint_angles = np.array(robot.get_joint_angles().msg)
            recorder.sample(start_time, joint_angles)
            joint_velocities = read_joint_velocities(robot)

            gravity_torque = pin.inverse_dynamics(
                joint_angles, joint_velocities, np.zeros_like(joint_velocities), R_world_base
            )
            apply_joint_torques(robot, gravity_torque)

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
        path = out_dir / f"joint_angles_{datetime.now():%Y%m%d_%H%M%S}.npz"
        recorder.save(path)
        print(f"saved {len(recorder)} samples to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=OUTPUT_DIR, help="directory to save in, created if missing"
    )
    main(parser.parse_args().out)
