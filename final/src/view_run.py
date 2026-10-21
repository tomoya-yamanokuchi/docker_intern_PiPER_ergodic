#!/usr/bin/env python3
"""Replay one trial of a finished run in MeshCat, and plot every trial's phase. Hardware-free.

The plot goes beside the run, as run_<...>.png. Trials count from 1.

Run from final/src:  sh run.sh view_run.py run_<method>_<ts>.npz [trial]
"""

import sys
from pathlib import Path

import numpy as np

from method.task import Task
from visualization.views import plot_run, show_run

path = Path(sys.argv[1])
run = dict(np.load(path))
plot_run(run, path.with_suffix(".png"))
print(f"outcomes: {list(run['outcomes'])}; plot in {path.with_suffix('.png')}")
show_run(Task.load(Path(str(run["task"]))), run, int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 0)
