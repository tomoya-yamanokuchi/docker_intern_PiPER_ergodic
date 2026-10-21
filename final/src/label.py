#!/usr/bin/env python3
"""Label: the master and the exploration distribution, turned into the task the law runs on.

Hardware-free. Every pose goes into the pose metric (rotation from the master's first
orientation, times ROTATION_LENGTH), the master is resampled at equal arc length, and each
datapoint gets the phase of its nearest master sample. Writes task_<timestamp>.npz to
final/output; view it with view_labels.py.

Run from final/src:  sh run.sh label.py master_<ts>.npz exploration_<ts>.npz
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from core.agx_pinocchio import AgxPinocchio
from method.pose import to_metric
from method.task import label
from robot import OUTPUT_DIR, URDF_PATH, tool_poses


def main(master: Path, exploration: Path, out_dir: Path = OUTPUT_DIR) -> Path:
    pin = AgxPinocchio(str(URDF_PATH))
    q_m, q_p = np.load(master)["q"], np.load(exploration)["q"]
    p_m, R_m = tool_poses(pin, q_m)
    p_p, R_p = tool_poses(pin, q_p)
    R0 = R_m[0]
    task = label(to_metric(p_m, R_m, R0), q_m, to_metric(p_p, R_p, R0), q_p, R0)
    print(
        f"master {task.S_m:.3f} m in the metric; {len(q_p)} datapoints; h = {task.h:.4f} m; "
        f"one resolvable width {task.sigma_f:.3f} in phase\ncube sides {np.round(task.L, 3)} m"
    )
    print(
        f"datapoints per tenth of the phase: {np.histogram(task.phi_p, np.linspace(0, 1, 11))[0]}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"task_{datetime.now():%Y%m%d_%H%M%S}.npz"
    task.save(path)
    print(f"saved {path}")
    return path


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
