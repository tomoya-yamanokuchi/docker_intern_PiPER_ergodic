#!/usr/bin/env python3
"""PreToolUse guard for Bash: refuse to run Python that can reach the arm (CLAUDE.md hard rule 1).

Every Python run here goes through `docker exec piper-ergodic bash -lc '...'`, which no
permission pattern can see into, so this reads the command itself. A command that runs
Python is refused if its own text, or any .py file it names, mentions a module that
touches the arm. Reading, grepping and editing those files runs no Python and passes.
Only one level is read: a named script that imports a helper that imports run_trials
is not caught.
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONTAINER_REPO = "/home/jens/workspace/docker_intern_PiPER_ergodic"
# The fakes and the offline chain that drives run_trials on them: safe by CLAUDE.md.
SAFE_FILES = {"simulate.py", "fake.py", "fake_executor_helpers.py"}

RUNS_PYTHON = re.compile(r"\b(i?python3?(\.\d+)?|run\.sh)\b")
TOUCHES_ARM = re.compile(
    r"\bpyAgxArm\b|\bexecutor_helpers\b|\barm[./]piper\b|\barm[./]control\b"
    r"|\brun_trials\b|\brecord_master\b|\bteach_exploration\b"
    r"|\brecord_joint_angles\b|\bteach_datapoints\b|\bplay_joint_angles\b|\bidentify_friction\b"
    r"|\brun_ergodic_\w+|\bmain_(gc|tast_imp|jnt_imp)\b"
    r"|\bmove_(mit|p|j)\s*\(|\brobot\.(connect|enable)\s*\("
)


def on_host(path: str) -> str:
    return path.replace(CONTAINER_REPO, str(REPO))


def named_files(command: str) -> list[Path]:
    """The .py files the command names, on the host; relative names are tried from every cd."""
    dirs = [Path.cwd()] + [Path(on_host(d)) for d in re.findall(r"\bcd\s+([^\s;&|']+)", command)]
    candidates = [
        base / on_host(n) for n in re.findall(r"[\w./-]+\.py\b", command) for base in dirs
    ]
    return list(dict.fromkeys(p for p in candidates if p.is_file()))


def main() -> None:
    command = json.load(sys.stdin).get("tool_input", {}).get("command", "")
    if not RUNS_PYTHON.search(command):
        return
    hits = [("the command", m.group()) for m in [TOUCHES_ARM.search(command)] if m]
    for path in named_files(command):
        m = TOUCHES_ARM.search(path.read_text(errors="replace"))
        if m and path.name not in SAFE_FILES:
            hits.append((str(path), m.group()))
    if hits:
        where = "; ".join(f"{w} names {what!r}" for w, what in hits)
        print(
            f"Blocked by CLAUDE.md hard rule 1: this runs Python that can reach the arm ({where}). "
            "The user runs arm scripts. For offline work, import the hardware-free modules "
            "(robot, method/, visualization/) instead; to edit such a file, use Edit or Write.",
            file=sys.stderr,
        )
        sys.exit(2)


main()
