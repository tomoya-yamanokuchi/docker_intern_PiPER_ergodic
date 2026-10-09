#!/usr/bin/env python3
"""Replay recorded phase runs through the phase estimator and the stall counter.

Hardware-free. Both are functions of the measured pose sequence alone, so feeding
the q of an ergodic_run_*.npz back through them is exact, and a constant can be
switched off on the arm's own data. The target and the statistic are closed-loop
and are not judged here.

A trace recorded before the estimator last changed will not match: the replay runs
today's estimator.

The runs are one session's, in order: the law's phase clock restarts at each. With
--trace the replayed phi is first checked against the phase_trace_*.npz the session
wrote, which is what shows the replay is the run.

Run from workspace/src:
    python replay_phase_runs.py master.npz datapoints_phase.npz run1.npz run2.npz --trace phase_trace.npz
"""

import argparse
import importlib
import time
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from direct_teaching.distribution.phase_projection import (
    phase_context,
    pose_distance,
    project,
    rotation_angle,
)
from ergodic_controller import phase_ergodic_controller
from ergodic_controller.phase_ergodic_controller import PhaseErgodicController
from simulation import fake_executor_helpers

# (label, PHASE_WINDOW): the estimator as it runs, and with its window switched off.
VARIANTS = (("as run", phase_ergodic_controller.PHASE_WINDOW), ("no window", (1.0, 1.0)))


def replay(
    law: PhaseErgodicController,
    states: list[np.ndarray],  # per run, (n, axes) cube states at the ergodic steps
) -> np.ndarray:  # (steps, 3): phi, stall, stall / T(phi)
    rows = []
    for X in states:
        law.phi, law.stall, law.last_progress, law.lag = 0.0, 0, 0.0, 0.0
        for x in X:
            law._advance_phase(x)
            rows.append((law.phi, law.stall, law.stall / law.dwell(law.phi)))
    return np.array(rows)


def report_labelling(master: Path, labelled: Path, pin_model) -> None:
    with np.load(labelled) as data:
        q, phi = data["q"], data["phi"]
        length = float(data["rotation_length"]) if "rotation_length" in data else 0.0
    context = phase_context(master, q, pin_model, length)
    d = pose_distance(
        context.p[:, None],
        context.quat[:, None],
        context.p_master[None],
        context.quat_master[None],
        length,
    ).min(axis=1)
    print(f"labelling: {len(q)} datapoints, l = {length} m/rad, h = {1000 * context.h:.1f} mm")
    print(
        f"  distance to the master: median {1000 * np.median(d):.1f} mm, max {1000 * d.max():.1f} mm, "
        f"{np.mean(d > context.h):.0%} further than h"
    )
    d_p = np.linalg.norm(context.p[:, None] - context.p_master[None], axis=2).min(axis=1)
    angle = rotation_angle(context.quat[:, None], context.quat_master[None]).min(axis=1)
    to_goal = np.linalg.norm(context.p_master - context.p_master[-1], axis=1)
    print(
        f"  nearest master position median {1000 * np.median(d_p):.1f} mm, "
        f"nearest master orientation median {np.median(angle):.2f} rad\n"
        f"  the master is within h of its last position from phi "
        f"{context.phi_master[np.argmax(to_goal < context.h)]:.2f}"
    )
    for other in (0.0, 0.03, 0.06, 0.12):
        relabelled = project(phase_context(master, q, pin_model, other))
        print(
            f"  l = {other}: rank correlation with the file's labels {spearmanr(phi, relabelled).statistic:.3f}, "
            f"max |dphi| {np.abs(phi - relabelled).max():.3f}"
        )


def report_estimator(law: PhaseErgodicController, states: list[np.ndarray]) -> np.ndarray:
    """Each variant on the master's own states, then on the runs; returns the runs' rows as run."""
    task = law.task
    ends = np.cumsum([len(X) for X in states]) - 1
    baseline = None
    for label, window in VARIANTS:
        phase_ergodic_controller.PHASE_WINDOW = window
        on_master = replay(law, [task.X_master])[:, 0]
        rows = replay(law, states)
        baseline = rows if baseline is None else baseline
        print(
            f"estimator, {label}: on the master max |phi - its own| {np.abs(on_master - task.phi_master).max():.3f}; "
            f"runs end at phi {np.round(rows[ends, 0], 2).tolist()}, "
            f"max |dphi| to as run {np.abs(rows[:, 0] - baseline[:, 0]).max():.3f}, "
            f"stalled {np.mean(rows[:, 1] > 0):.0%} of steps"
        )
    phase_ergodic_controller.PHASE_WINDOW = VARIANTS[0][1]
    return baseline


def report_stall(rows: np.ndarray, error: np.ndarray) -> None:
    """error (steps,) m, commanded minus measured peg tip: a jam signal the counter never sees."""
    stalled = rows[:, 1] > 0
    if stalled.all() or not stalled.any():
        print(f"stall: {'every' if stalled.all() else 'no'} step stalled, nothing to compare")
        return
    auroc = mannwhitneyu(error[stalled], error[~stalled]).statistic / (
        stalled.sum() * (~stalled).sum()
    )
    print(
        f"stall: tracking error median {1000 * np.median(error[stalled]):.2f} mm stalled, "
        f"{1000 * np.median(error[~stalled]):.2f} mm not; AUROC {auroc:.2f} (0.5 is unrelated)"
    )


def report_history(
    law: PhaseErgodicController,
    states: list[np.ndarray],
    rows: np.ndarray,
    dt: float,  # s, the ergodic step period
) -> None:
    """What run_ergodic_phase_trials.forget_progress keeps per run, and what a step then costs."""
    ends = np.cumsum([len(X) for X in states])
    kept = [
        int((rows[a:b, 2] >= 1.0).sum()) for a, b in zip(np.r_[0, ends[:-1]], ends, strict=True)
    ]
    print(f"history: states kept per run {kept}, of {[len(X) for X in states]} recorded")
    X = np.vstack(states)
    for n in (1000, 2000, 4000, 8000):
        law.memory_x, law.memory_phi = (
            list(X[np.arange(n) % len(X)]),
            list(rows[np.arange(n) % len(X), 0]),
        )
        law._cached = -1
        start = time.perf_counter()
        for _ in range(5):
            law.step(X[0], dt)
        ms = 1000 * (time.perf_counter() - start) / 5
        print(f"  {n} remembered states: {ms:.1f} ms a step, a new state weighs 1/{n}")


def main(args: argparse.Namespace) -> None:
    # live_ergodic_controller holds the run's constants but imports the SDK boundary.
    fake_executor_helpers.install(np.zeros(6))
    live = importlib.import_module("execution.live_ergodic_controller")
    controller, distribution, law, _ = live.prepare_phase_law(
        args.master, args.labelled, planar=not args.six_dof, track_master=True
    )
    fk = controller.pin_model.forward_kinematics
    states, errors = [], []
    for path in args.runs:
        with np.load(path) as run:
            # The law stepped on cycle 0 and every cycles_per_step-th after it.
            every = round(live.CONTROL_FREQ_HZ / live.ERGODIC_FREQ_HZ)
            q, p_target = run["q"][::every], run["p_target"][::every]
        poses = [fk(q_i, live.TCP_FRAME_NAME) for q_i in q]
        X = np.clip([distribution.pose_to_state(p, R) for p, R in poses], 0.0, 1.0)
        states.append(X[:, : law.task.axes])
        errors.append(np.linalg.norm(p_target - np.array([p for p, _ in poses]), axis=1))

    report_labelling(args.master, args.labelled, controller.pin_model)
    rows = report_estimator(law, states)
    if args.trace:
        recorded = np.load(args.trace)["phi"]
        n = min(len(recorded), len(rows))
        print(
            f"self-check: replayed phi against {args.trace.name}, {len(rows)} against {len(recorded)} steps, "
            f"max |dphi| {np.abs(rows[:n, 0] - recorded[:n]).max():.4f}"
        )
    report_stall(rows, np.concatenate(errors))
    report_history(law, states, rows, 1.0 / live.ERGODIC_FREQ_HZ)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("master", type=Path, help="joint_angles_*.npz, the master trajectory")
    parser.add_argument("labelled", type=Path, help="datapoints_*_phase.npz the runs used")
    parser.add_argument(
        "runs", type=Path, nargs="+", help="ergodic_run_*.npz, one session in order"
    )
    parser.add_argument("--trace", type=Path, help="phase_trace_*.npz of the same session")
    parser.add_argument("--six-dof", action="store_true", help="the runs used the 6-DoF law")
    main(parser.parse_args())
