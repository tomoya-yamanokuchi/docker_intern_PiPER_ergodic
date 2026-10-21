#!/usr/bin/env python3
"""The labelled datapoints, and the sliding target's weights along the phase. Hardware-free.

Run from final/src:  sh run.sh view_labels.py task_<ts>.npz
"""

import sys
from pathlib import Path

from method.task import Task
from visualization.views import show_labels

show_labels(Task.load(Path(sys.argv[1])))
