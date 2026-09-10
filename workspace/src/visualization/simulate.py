#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Travel between two poses with no hardware attached, and stream it to the viewer.

The question this answers is "would this trajectory work": is the target
reachable, does every joint stay inside its limits, and does the path swept
between the two poses go anywhere it should not. You watch the answer on the
visualised arm.

    python -m visualization watch                                  # shell 1
    python -m visualization simulate --start zero --target ready   # shell 2

What is exercised is the actual control path, not a re-implementation of it:
move_to() plans the same smoothstep trajectory, compose_mit_targets() attaches
the same gains and check_targets() applies the same joint limits, against
SimulatedArm in place of the CAN transport.

Run it without a viewer and it still does useful work -- an unreachable angle is
rejected here exactly as it would be on hardware, and the tracking report prints
the path it took.
"""

import numpy as np
from impedance_control.motion import MotionProfile, move_to
from impedance_control.telemetry import open_telemetry
from kinematics.dynamics import make_tau_ff_fn
from kinematics.model import Q_READY

from visualization.simulated_arm import SimulatedArm

JOINTS = [1, 2, 3, 4, 5, 6]

# Poses that can be named on the command line instead of typed out.
NAMED_POSES = {"zero": np.zeros(6), "ready": np.asarray(Q_READY, dtype=float)}

# Gains are carried through the real code path and validated by check_targets(),
# but they do not shape the motion here: SimulatedArm tracks perfectly, so what
# is being checked is the geometry of the path, not the impedance behaviour.
TUNING = {joint: {"qdot_ref": 0.0, "kp": 5.0, "kd": 0.8, "tau_ff": 0.0} for joint in JOINTS}


def parse_pose(text, kin):
    """
    A pose given on the command line, as a {joint: q} dict in rad.

    Either a name from NAMED_POSES, or six comma-separated joint angles in
    radians, j1 first. Rejected here rather than deep inside the trajectory if
    it is the wrong length or outside the joint limits, so a typo says so before
    anything is drawn.
    """
    if text in NAMED_POSES:
        q = NAMED_POSES[text]
    else:
        values = [float(v) for v in text.replace(" ", "").split(",")]

        if len(values) != 6:
            raise ValueError(
                f"a pose is {', '.join(NAMED_POSES)} or six comma-separated "
                f"joint angles in rad, got {len(values)} numbers in '{text}'"
            )

        q = np.array(values)

    if not kin.in_limits(q):
        outside = [
            f"j{joint}: {q[joint - 1]:.4f} outside "
            f"[{kin.q_min[joint - 1]:.4f}, {kin.q_max[joint - 1]:.4f}]"
            for joint in JOINTS
            if not kin.q_min[joint - 1] <= q[joint - 1] <= kin.q_max[joint - 1]
        ]

        raise ValueError(f"pose '{text}' is outside the joint limits -- {'; '.join(outside)}")

    return {joint: float(q[joint - 1]) for joint in JOINTS}


def simulate_trajectory(kin, args):
    """
    Put the simulated arm at --start, travel to --target, streaming every step.
    """
    q_start = parse_pose(args.start, kin)
    q_target = parse_pose(args.target, kin)

    arm = SimulatedArm(kin, q_start=[q_start[joint] for joint in JOINTS])

    profile = MotionProfile(
        tuning=TUNING,
        tau_ff_fn=make_tau_ff_fn(kin, 1.0, JOINTS),
        max_speed=args.max_speed,
        rate=args.rate,
        telemetry=open_telemetry(),
    )

    print("INFO: simulating with no hardware; `python -m visualization watch` to see it")
    print("INFO: start ", {joint: round(q, 4) for joint, q in q_start.items()})
    print("INFO: target", {joint: round(q, 4) for joint, q in q_target.items()})
    print("")

    # arm stands in for both the Piper handle and the CAN interface.
    move_to(arm, arm, q_target, profile)

    print("")
    print(f"INFO: done, {len(arm.commands)} frames would have gone to the arm")
