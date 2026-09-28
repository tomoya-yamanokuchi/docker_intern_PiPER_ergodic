#!/usr/bin/env python3
"""The E2T2 trialled peg-in-hole experiment: N insertions, one accumulated statistic.

This is the experiment of Fig. 6 in Shetty, Silverio & Calinon, "Ergodic Exploration
using Tensor Train" (arXiv 2101.04428, section VI-A), run on the arm. The paper:

    as soon as the system reaches the boundary of the target region, we re-initialize
    it to the initial state and repeat this process for a fixed number of times [...]
    the cumulative average decreases with the number of successful attempts and
    converges to a fixed value

Three things carry the result, and all three are decisions this script makes:

1. **Only the arm's pose is reset between trials; the statistic is not.** One
   ErgodicController is built before the first trial and every trial steps that same
   object, so its accumulated time-spatial characteristic W -- the running sum of
   Phi(x) in tensor-train format, with step_count as the normaliser of eq. (3) --
   keeps growing across trial boundaries. Clearing it, or building a controller per
   trial, deletes the effect the experiment exists to measure.
2. **Every trial starts from the same pose**, the one the arm was in when the
   sequence began. That is what gives the curve a floor: the time to travel from
   that pose straight to the hole. The placement prompt reports the error against it.
3. **The statistic does not accumulate while the operator repositions the arm.**
   The paper's re-initialisation is instantaneous; ours is a human moving the arm
   under gravity compensation, and feeding that path to the law would mark whatever
   it was dragged through as explored.

One trial, in the terminal: the arm explores until the peg tip enters the goal
sphere, GOAL_RADIUS around GOAL_TCP, and the trial ends on that cycle. The arm then
goes backdrivable, you place it at the start pose, and Enter begins the next trial.
Ctrl-C at any point holds the arm and keeps the trials that finished.

Interactive only. Enter is polled with select, so under a redirected stdin EOF reads
as an endless stream of newlines and the placement phase ends on its first cycle.

Run from workspace/src:  python run_ergodic_trials.py recording.npz 5
"""

import argparse
import select
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tt

from ergodic_controller.ergodic_controller import ErgodicController
from execution.executor_helpers import (
    apply_joint_torques,
    connect_arm,
    hold_current_pose,
    read_joint_velocities,
)
from execution.live_ergodic_controller import (
    CONTROL_FREQ_HZ,
    OUTPUT_DIR,
    R_WORLD_BASE,
    TCP_FRAME_NAME,
    Exploration,
    explore,
    first_setpoints,
    prepare_exploration,
    print_start,
    save_run,
)
from visualization.visualizer import cumulative_average, save_trial_times_plot, trials_accepted

PLACEMENT_PRINT_CYCLES = 50  # at CONTROL_FREQ_HZ, a placement line every 0.5 s

# The hole, as a sphere the peg tip has to reach for a trial to count as a success.
# The centre is the mean peg_tcp position over the ten confirmed insertions of
# 2026-09-25, which all sit within 5.4 mm of it; the radius covers that spread with
# margin. Both are bench geometry: re-measure them if the fixture moves.
GOAL_TCP = np.array([0.28276, 0.00599, 0.00373])  # m, in the base frame
GOAL_RADIUS = 0.005  # m


def enter_pressed() -> bool:
    """True once a line is waiting on stdin, consuming it.

    Polled rather than read, for the reason teach_datapoints.read_command gives:
    blocking on input() would stop the torque stream, and the firmware latches the
    last torque, so the arm would hold a stale one instead of tracking or
    compensating.
    """
    if not select.select([sys.stdin], [], [], 0)[0]:
        return False
    sys.stdin.readline()
    return True


def drain_stdin() -> None:
    """Discard anything typed while the arm was exploring, before placing starts.

    Enter ends the placement phase, so a newline typed during the trial -- out of
    habit, or to see whether the run is still alive -- would skip the placement
    silently and start trial k+1 from wherever the previous one ended, which
    corrupts the measurement rather than the arm.

    readline returns "" only at EOF, and that is what stops this loop: under a
    redirected stdin select reports readable forever, so testing the line rather
    than select is what keeps this from spinning.
    """
    while select.select([sys.stdin], [], [], 0)[0] and sys.stdin.readline() != "":
        pass


def reached_goal(exploration: Exploration, q: np.ndarray) -> bool:
    """True once the peg tip is inside the goal sphere.

    Position only: the hole constrains the peg's orientation mechanically, so a tip
    within GOAL_RADIUS of GOAL_TCP is an insertion. Called once per control cycle,
    which is one peg_tcp forward kinematics on top of the cycle's own work.
    """
    p, _ = exploration.controller.pin_model.forward_kinematics(q, TCP_FRAME_NAME)
    return bool(np.linalg.norm(p - GOAL_TCP) <= GOAL_RADIUS)


def print_placement(
    exploration: Exploration,
    q: np.ndarray,  # (6,) rad, where the arm is now
    q_start: np.ndarray,  # (6,) rad, the pose every trial starts from
    trial: int,
) -> None:
    """How far the arm still is from the pose every trial starts at."""
    pin_model = exploration.controller.pin_model
    p, _ = pin_model.forward_kinematics(q, TCP_FRAME_NAME)
    p_start, _ = pin_model.forward_kinematics(q_start, TCP_FRAME_NAME)
    print(
        f"place the arm for trial {trial}: max joint error {np.max(np.abs(q - q_start)):.4f} rad, "
        f"TCP {np.linalg.norm(p - p_start):.4f} m; Enter to start"
    )


def backdrive_to_start(
    robot,
    exploration: Exploration,
    q_start: np.ndarray,  # (6,) rad, the pose every trial starts from
    trial: int,
) -> np.ndarray:  # (6,) rad, the pose the arm was left in
    """Gravity compensation so the operator can put the arm back at q_start.

    The record_joint_angles.py loop: inverse dynamics with the measured velocity and
    zero acceleration, so gravity and the Coriolis terms are both carried and the arm
    is backdrivable.

    The ergodic controller is deliberately not stepped here. Its statistic is meant
    to be the coverage exploration achieved, and the operator's handling path is not
    that -- dragging the peg across the hole on the way back would mark the hole as
    visited and slow the next trial down, which is exactly the quantity being
    measured.
    """
    drain_stdin()
    pin_model = exploration.controller.pin_model
    period = 1.0 / CONTROL_FREQ_HZ
    cycle = -1
    while True:
        cycle += 1
        start_time = time.monotonic()
        joint_angles = np.array(robot.get_joint_angles().msg)
        joint_velocities = read_joint_velocities(robot)
        apply_joint_torques(
            robot,
            pin_model.inverse_dynamics(
                joint_angles, joint_velocities, np.zeros_like(joint_velocities), R_WORLD_BASE
            ),
        )
        if enter_pressed():
            return joint_angles
        if cycle % PLACEMENT_PRINT_CYCLES == 0:
            print_placement(exploration, joint_angles, q_start, trial)

        elapsed_time = time.monotonic() - start_time
        if elapsed_time < period:
            time.sleep(period - elapsed_time)
        else:
            print(f"warning: control loop overrun {elapsed_time:.3f}s > {period:.3f}s")


def snapshot_statistics(ergodic: ErgodicController) -> tuple[list[np.ndarray], int]:
    """The accumulated time-spatial characteristic W as TT cores, and its normaliser.

    Copied out of the live tensor, because the controller replaces tt_wt whenever it
    rounds. tt_wt is the running *sum* of Phi(x), so the paper's time-averaged
    statistic of eq. (3) is this divided by step_count: the cores are meaningless
    without the count, which is why the two are always stored together.
    """
    return [np.array(core) for core in tt.vector.to_list(ergodic.tt_wt)], ergodic.step_count


def run_one_trial(
    robot,
    exploration: Exploration,
    recording: Path,
    q: np.ndarray,  # (6,) rad, the pose this trial starts from
    trial: int,
) -> float:  # s, from the first torque to the peg entering the goal sphere
    """Explore until the peg tip reaches the hole.

    The per-cycle log is written whatever happens, so a trial abandoned with Ctrl-C
    or ended by a torque-limit refusal still leaves its trajectory on disk in the
    shape visualization.visualizer.show_ergodic_run reads.
    """
    setpoints = first_setpoints(exploration, q)
    print_start(exploration, q, setpoints)
    print(f"exploring; the trial ends when the peg reaches {np.round(GOAL_TCP, 4)} m")
    # (t s, q (6,) rad, p_target (3,) m, R_target (3, 3), tau (6,) N*m) per cycle
    log = []
    try:
        return explore(
            robot, exploration, setpoints, log, stop=lambda q: reached_goal(exploration, q)
        )
    finally:
        if log:
            save_run(log, recording, label=f"_trial{trial}")


def save_trials(path: Path, recording: Path, trials: list[tuple]) -> None:
    """One file for the whole sequence: the times, the start poses and W after each trial.

    Reload the statistic after trial k, counting from zero, with

        tt.vector.from_list([data[f"w{k}_c{i}"] for i in range(6)])

    and divide by step_counts[k] for the time average of eq. (3). The recording is
    stored so the distribution W is compared against can be refitted offline.
    """
    seconds, q_start, cores, step_counts = zip(*trials, strict=True)
    w = {
        f"w{trial}_c{i}": core
        for trial, snapshot in enumerate(cores)
        for i, core in enumerate(snapshot)
    }
    np.savez(
        path,
        recording=str(recording),
        trial_times=np.array(seconds),
        q_start=np.array(q_start),
        step_counts=np.array(step_counts),
        **w,
    )
    print(f"saved {len(seconds)} trials to {path}")


def report_trials(path: Path, trials: list[tuple]) -> None:
    """The per-trial table, the verdict, and the figure beside the data file."""
    if not trials:
        print("no trial finished, nothing to report")
        return
    trial_times = np.array([seconds for seconds, *_ in trials])
    averages = cumulative_average(trial_times)
    print("\ntrial   time [s]   cumulative average [s]")
    for trial, (seconds, average) in enumerate(zip(trial_times, averages, strict=True), start=1):
        print(f"{trial:5d}   {seconds:8.1f}   {average:22.1f}")
    verdict = "accepted" if trials_accepted(trial_times) else "not accepted"
    print(f"cumulative average {averages[0]:.1f} s -> {averages[-1]:.1f} s: {verdict}")
    figure = path.with_suffix(".png")
    save_trial_times_plot(trial_times, figure)
    print(f"saved {figure}")


def run_trials(robot, exploration: Exploration, recording: Path, n_trials: int) -> None:
    """The sequence, and the arm for its whole duration.

    q_start is read once. Every later trial is placed back to it rather than started
    from wherever the previous one ended, so the times differ by how the exploration
    improved and not by how far the arm had to travel.
    """
    path = OUTPUT_DIR / f"ergodic_trials_{datetime.now():%Y%m%d_%H%M%S}.npz"
    # (s to success, start q (6,) rad, W cores, ergodic step count) per finished trial
    trials: list[tuple] = []
    q = q_start = np.array(robot.get_joint_angles().msg)
    try:
        for trial in range(1, n_trials + 1):
            if trial > 1:
                q = backdrive_to_start(robot, exploration, q_start, trial)
            seconds = run_one_trial(robot, exploration, recording, q, trial)
            trials.append((seconds, q, *snapshot_statistics(exploration.ergodic)))
            average = cumulative_average([s for s, *_ in trials])[-1]
            print(
                f"trial {trial} of {n_trials}: {seconds:.1f} s, cumulative average {average:.1f} s"
            )
    except KeyboardInterrupt:
        print("\ninterrupted, holding current pose")
    finally:
        # Read the arm rather than reuse a q from the loop: an interrupt during
        # backdriving arrives while the operator is holding the arm somewhere the
        # loop never saw, and handing that stale pose to the driver's PD would snap
        # the arm back to it. The SDK returns None until it has a valid frame, so
        # fall back rather than raise here and leave an arm with no brakes unheld.
        measured = robot.get_joint_angles()
        hold_current_pose(robot, np.array(measured.msg) if measured is not None else q)
        if trials:
            save_trials(path, recording, trials)
    report_trials(path, trials)


def main(recording: Path, n_trials: int) -> None:
    exploration = prepare_exploration(recording)
    robot = connect_arm()
    run_trials(robot, exploration, recording, n_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # Mandatory: the arm explores whatever recording this is, so never pick one implicitly.
    parser.add_argument("recording", type=Path, help="joint_angles_*.npz or datapoints_*.npz")
    parser.add_argument("n_trials", type=int, help="insertion attempts to run")
    args = parser.parse_args()
    main(args.recording, args.n_trials)
