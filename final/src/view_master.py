#!/usr/bin/env python3
"""The arm along a recorded master, coloured and keyed by phase. Hardware-free.

Run from final/src:  sh run.sh view_master.py master_<ts>.npz
"""

import sys

import numpy as np

from core.agx_pinocchio import AgxPinocchio
from method.pose import to_metric
from method.task import resample, trim_dwell
from robot import URDF_PATH, tool_poses
from visualization.views import show_master

q = np.load(sys.argv[1])["q"]
p, rot = tool_poses(AgxPinocchio(str(URDF_PATH)), q)
y = to_metric(p, rot, rot[0])
keep = trim_dwell(y)
show_master(*resample(y[keep], q[keep]))
