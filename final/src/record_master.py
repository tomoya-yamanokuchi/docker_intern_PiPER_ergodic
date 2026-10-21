#!/usr/bin/env python3
"""Record the master: one demonstration of the whole task, backdriven under gravity compensation.

TOUCHES THE ARM. Every control cycle's q is kept, at 100 Hz. Ctrl-C hands the arm to a
position hold, then writes master_<timestamp>.npz (t (n,) s, q (n, 6) rad) to final/output.
The dwell at either end is trimmed when the master is labelled, so there is no hurry.

Run from final/src:  sh run.sh record_master.py
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np

from arm.control import gravity_compensation, keep_rate
from arm.piper import connect_arm, hold_current_pose
from core.agx_pinocchio import AgxPinocchio
from robot import OUTPUT_DIR, URDF_PATH


def main(robot, out_dir: Path = OUTPUT_DIR) -> Path:
    pin = AgxPinocchio(str(URDF_PATH))
    t, q = [], []
    print("gravity compensation on, recording; Ctrl-C stops and saves")
    t0 = time.monotonic()
    try:
        while True:
            start = time.monotonic()
            q.append(gravity_compensation(robot, pin))
            t.append(start - t0)
            keep_rate(start)
    except KeyboardInterrupt:
        print("\ninterrupted, holding the current pose")
    finally:
        # Hold before any file I/O, so a failed write cannot leave the arm unheld.
        hold_current_pose(robot, np.array(robot.get_joint_angles().msg))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"master_{datetime.now():%Y%m%d_%H%M%S}.npz"
    np.savez(path, t=np.array(t), q=np.array(q))
    print(f"saved {len(q)} samples to {path}")
    return path


if __name__ == "__main__":
    main(connect_arm())
