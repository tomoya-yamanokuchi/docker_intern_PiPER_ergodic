#!/usr/bin/env python3
"""Identify the joint friction of the arm, for the FeedForward of play_joint_angles.py.

At each of POSE_COUNT poses it measures every joint's breakaway torque, then
sweeps the joints one at a time back and forth at several constant speeds under
the joint impedance law of main_jnt_imp.py. The operator backdrives the arm to
each pose by hand under gravity compensation; the script waits for Enter.

Friction comes out of the torque balance of every logged sample,

    f = tau - nle(q, qd) - M(q) qdd,

binned by the MEASURED speed. That identity holds whether or not the joint is
tracking its target, which matters here: asked for 0.05 rad/s the joints do not
slide at all, they stick and then lurch at three times the command, so averaging
over a band of the COMMANDED speed -- what this script used to do -- reads
stick-slip transients rather than a sliding level. Binning the measured speed
turns those same transients into the low-speed end of a friction curve.

The sweeps run with the friction of controller/feed_forward.py already fed
forward, so the joints track better than they otherwise would. The
identification is unaffected by that: the balance above returns the friction
itself, however the command was composed.

Every speed gets the same wall clock, and the travel is cut at slow speeds so
each speed still reverses several times inside its window. With equal cycles per
speed instead, the slowest contributed 26 times the samples of the fastest and a
median over the pooled samples was whatever the slow sweep said.

At the first pose it also sweeps the dwell time: the same breakaway ramp on the
joints in DWELL_JOINTS after each of DWELL_TIMES at rest, rather than the one
second the other ramps get. Stiction grows with time at rest, and an arm running
a task stands still for far longer than a second, so this says whether
FRICTION_STATIC as measured here is the level the task meets or only a floor.

Ctrl-C hands the arm to a position hold and still fits whatever was logged.

Run from workspace/src:  python identify_friction.py
"""

import select
import sys
import time
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from controller.feed_forward import FRICTION_COULOMB, FRICTION_STATIC, FeedForward
from controller.jnt_imp_controller import JointImpedanceController
from execution.executor_helpers import (
    apply_joint_torques,
    connect_arm,
    hold_current_pose,
    read_joint_velocities,
)

AGX_REFERENCE = Path(__file__).resolve().parent / "agx_reference"
URDF_PATH = AGX_REFERENCE / "piper/piper/urdf/piper_description.urdf"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

CONTROL_FREQ_HZ = 200.0
# Sweeps are repeated at several poses because nothing so far says whether
# friction varies with configuration and load; every earlier run swept one pose.
POSE_COUNT = 2
SWEEP_AMPLITUDE = 0.3  # rad, either side of the pose, at the fastest speeds
# Log-spaced and slow-biased, because friction varies most near zero. 0.8 rad/s
# is gone: above roughly 0.4 the joints overshoot their triangle wave instead of
# tracking it -- joint 2 measured 1.17 rad/s for a commanded 0.8 -- and the
# friction read off that comes out negative, which is not a measurement.
SWEEP_SPEEDS = (0.02, 0.05, 0.1, 0.2, 0.4)  # rad/s
# Set from how fast the binned median converges: resampling the earlier
# recordings, 500 samples in a bin hold it inside 0.008-0.038 N*m, far under the
# spread between runs, and no bin any speed fills stays short of that in 8 s.
SWEEP_SECONDS = 8.0  # per speed, the same for every speed
SWEEP_HALF_PERIOD = 2.0  # s, one end of the travel to the other; sets the slow spans
LIMIT_MARGIN = 0.1  # rad, kept free at each joint limit

# Breakaway: the torque a joint needs before it moves at all, which the sweeps
# cannot see because it only exists while the joint is stopped. Slow enough that
# the joint stays quasi-static, and bounded well under the t_ff limits. The rate
# is not what sets the resolution: t_ff is quantised to 0.251 N*m on joints 1-3
# and 0.051 on 4-6, so what reaches the joint is a staircase and a slower ramp
# only lengthens its treads. At the 0.25 N*m/s this used to run at, a tread on
# joints 1-3 lasted a full second for no gain -- and a good part of the 19-50
# percent scatter between readings is that quantisation, not the gear mesh.
BREAKAWAY_RAMP_RATE = 1.0  # N*m/s
BREAKAWAY_MAX_TORQUE = 2.0  # N*m, above which the joint is called stuck
BREAKAWAY_MOTION = 0.01  # rad, counts as the joint having started to move
BREAKAWAY_SETTLE_S = 1.0  # s of plain hold, for the spring to take the joint back
# One ramp per direction sampled a quantity that scatters 19-50 percent between
# runs and by up to a factor of 2.5 between the two directions of one run, so a
# single reading is a draw from that spread rather than a measurement of it.
BREAKAWAY_REPEATS = 3  # per joint per direction

# Every reading above is of a joint that has been at rest for BREAKAWAY_SETTLE_S,
# one second. An arm running a task stands still far longer -- the ergodic runs
# of 2026-09-24 are motionless 73 percent of the time, in stretches of up to 11 s
# -- and stiction grows with time at rest. That matters because those same runs
# hold joint 3 still at 1.27 N*m against the 1.04 N*m breakaway measured here, so
# this script may be reporting a floor rather than the level the task meets. The
# sweep asks the same question at increasing dwells. The span is wider than the
# task's own stretches on purpose: the effect is logarithmic in time, so it needs
# nearly two decades to show a slope at all.
DWELL_TIMES = (1.0, 5.0, 20.0, 60.0)  # s at rest before the ramp
# 0-indexed: joints 3 and 5, the two the ergodic runs hold still above their
# measured breakaway. Joint 1's excess is already explained by its direction
# asymmetry, so it is not worth the minutes.
DWELL_JOINTS = (2, 4)

# Bin edges of the friction curve, log-spaced over the speeds the joints reach.
FRICTION_BIN_EDGES = (0.005, 0.01, 0.02, 0.04, 0.07, 0.12, 0.2, 0.35, 0.6)  # rad/s
FRICTION_BIN_MIN_SAMPLES = 50
# qd is quantised to 0.001 rad/s, so qdd from a raw difference is noise. The
# median over a bin absorbs what is left.
QDD_SMOOTH_SAMPLES = 5  # 25 ms at 200 Hz
# The Coulomb level is the median of the curve over this band; the static level
# comes from the breakaway ramps, which start from a genuine standstill. The band
# is the speed the task actually runs at, and it is also the fastest band where
# all six joints still read a physical level: above roughly 0.2 rad/s the curve
# of joints 4-6 keeps falling and crosses zero, which no friction does. Whatever
# causes that -- the 0.051 N*m torque quantum on those joints is the same size as
# the effect, and velocity may lag -- it is not something to calibrate against.
COULOMB_BAND = (0.07, 0.2)  # rad/s

# (pose, joint, q, qd, tau, qd_des) per control cycle
SweepLog = list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, float]]


def make_controller(dofs: int) -> JointImpedanceController:
    """JointImpedanceController with the gains validated in main_jnt_imp.py."""
    controller = JointImpedanceController(urdf_path=str(URDF_PATH), dofs=dofs)
    controller.set_jnt_params(
        b=np.array([0.5, 0.8, 0.8, 0.2, 0.2, 0.2]),
        k=np.array([10.0, 10.0, 10.0, 2.0, 1.0, 1.0]),
    )
    return controller


def sweep_range(
    q_start: float,  # rad
    joint_limits: tuple[float, float],  # rad
    speed: float,  # rad/s
) -> tuple[float, float]:  # rad
    """Travel of one joint at one speed, around q_start and inside its limits.

    The span is what the joint covers in SWEEP_HALF_PERIOD at this speed, capped
    at SWEEP_AMPLITUDE, so every speed reverses about as often inside its window.
    At a fixed span the slowest speed would creep one way for the whole of it and
    never sample the other direction, which is where the asymmetry lives.
    """
    amplitude = min(SWEEP_AMPLITUDE, 0.5 * speed * SWEEP_HALF_PERIOD)
    lower, upper = joint_limits
    return (
        max(q_start - amplitude, lower + LIMIT_MARGIN),
        min(q_start + amplitude, upper - LIMIT_MARGIN),
    )


def triangle_wave(
    t: float, low: float, high: float, speed: float, q_start: float
) -> tuple[float, float]:  # (rad, rad/s)
    """Position and velocity of a triangle wave between low and high, starting at q_start.

    Starting at the joint's own angle rather than at a corner keeps the impedance
    target continuous with the pose the arm is already holding.
    """
    span = high - low
    phase = (speed * t + q_start - low) % (2.0 * span)
    if phase < span:
        return low + phase, speed
    return high - (phase - span), -speed


def hold_pose(
    robot,
    controller: JointImpedanceController,
    q_hold: np.ndarray,  # (6,) rad
    seconds: float,
) -> np.ndarray:  # (6,) rad, the q of the last cycle
    """Stream the impedance hold with no extra torque, so a joint settles back onto q_hold.

    A plain sleep would not do: the arm has no watchdog and latches the last
    torque it was sent, so pausing after a ramp would leave the ramp driving.
    """
    period = 1.0 / CONTROL_FREQ_HZ
    zero = np.zeros_like(q_hold)
    joint_angles = q_hold
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        start_time = time.monotonic()
        joint_angles, _, _ = run_control_cycle(robot, controller, q_hold, zero, zero)
        elapsed_time = time.monotonic() - start_time
        if elapsed_time < period:
            time.sleep(period - elapsed_time)
    return joint_angles


def measure_breakaway(
    robot,
    controller: JointImpedanceController,
    joint: int,  # 0-indexed
    q_hold: np.ndarray,  # (6,) rad
    direction: float,  # +1 or -1
) -> tuple[float, np.ndarray]:  # N*m at first motion (nan if none), q of the last cycle
    """Ramp a torque on one joint until it moves; that torque is its static friction.

    The impedance spring keeps holding the joint throughout, so once it breaks
    free it travels only until the spring takes up the excess over the friction,
    a few degrees at most, and then stops on its own.
    """
    period = 1.0 / CONTROL_FREQ_HZ
    q_ramp_start = np.array(robot.get_joint_angles().msg)
    zero = np.zeros_like(q_hold)
    extra_torque = np.zeros_like(q_hold)
    joint_angles = q_ramp_start

    t0 = time.monotonic()
    while True:
        start_time = time.monotonic()
        ramp_torque = BREAKAWAY_RAMP_RATE * (start_time - t0)
        if ramp_torque > BREAKAWAY_MAX_TORQUE:
            return float("nan"), joint_angles

        extra_torque[joint] = direction * ramp_torque
        joint_angles, _, _ = run_control_cycle(robot, controller, q_hold, zero, extra_torque)
        if abs(joint_angles[joint] - q_ramp_start[joint]) > BREAKAWAY_MOTION:
            return ramp_torque, joint_angles

        elapsed_time = time.monotonic() - start_time
        if elapsed_time < period:
            time.sleep(period - elapsed_time)


def measure_all_breakaway(
    robot,
    controller: JointImpedanceController,
    q_hold: np.ndarray,  # (6,) rad
) -> tuple[np.ndarray, np.ndarray]:  # (6, 2, REPEATS) N*m for (-, +), q of the last cycle
    """Static friction of every joint, both directions, BREAKAWAY_REPEATS times each.

    The directions stay in separate columns rather than being averaged: one run
    read 1.163 N*m against joint 1 one way and 0.461 the other, and folding that
    into a single number hides it.
    """
    breakaway = np.full((robot.joint_nums, 2, BREAKAWAY_REPEATS), np.nan)
    joint_angles = q_hold
    q_target = q_hold.copy()
    for joint in range(robot.joint_nums):
        lower, upper = robot.get_config()["joint_limits"][f"joint{joint + 1}"]
        for column, direction in enumerate((-1.0, 1.0)):
            for repeat in range(BREAKAWAY_REPEATS):
                # Re-aim the spring at wherever the joint now is, before settling
                # on it. After a breakaway the joint coasts and stops where the
                # spring balances its static friction, and the spring is then
                # wound up in the opposite direction -- which would pay for most
                # of the next ramp and make the next reading far too low. Each
                # repeat therefore starts from a slightly different angle, which
                # is what makes the repeats sample the scatter instead of
                # repeating one gear mesh; the clip stops that walk at a limit.
                q_target[joint] = np.clip(
                    joint_angles[joint], lower + LIMIT_MARGIN, upper - LIMIT_MARGIN
                )
                joint_angles = hold_pose(robot, controller, q_target, BREAKAWAY_SETTLE_S)
                breakaway[joint, column, repeat], joint_angles = measure_breakaway(
                    robot, controller, joint, q_target, direction
                )
            sign = "+" if direction > 0 else "-"
            print(f"joint {joint + 1} {sign}: {np.round(breakaway[joint, column], 3).tolist()} N*m")
    return breakaway, joint_angles


def measure_dwell_breakaway(
    robot,
    controller: JointImpedanceController,
    q_hold: np.ndarray,  # (6,) rad
) -> tuple[np.ndarray, np.ndarray]:  # (DWELL_JOINTS, 2, DWELL_TIMES) N*m, q of the last cycle
    """Breakaway of DWELL_JOINTS after each of DWELL_TIMES at rest, both directions.

    The same ramp as measure_all_breakaway, differing only in how long the joint
    is held still first, so the two are directly comparable at DWELL_TIMES[0].
    """
    dwell = np.full((len(DWELL_JOINTS), 2, len(DWELL_TIMES)), np.nan)
    joint_angles = q_hold
    q_target = q_hold.copy()
    for row, joint in enumerate(DWELL_JOINTS):
        lower, upper = robot.get_config()["joint_limits"][f"joint{joint + 1}"]
        for column, direction in enumerate((-1.0, 1.0)):
            for k, seconds in enumerate(DWELL_TIMES):
                q_target[joint] = np.clip(
                    joint_angles[joint], lower + LIMIT_MARGIN, upper - LIMIT_MARGIN
                )
                joint_angles = hold_pose(robot, controller, q_target, seconds)
                dwell[row, column, k], joint_angles = measure_breakaway(
                    robot, controller, joint, q_target, direction
                )
            sign = "+" if direction > 0 else "-"
            print(
                f"joint {joint + 1} {sign}: after {list(DWELL_TIMES)} s at rest -> "
                f"{np.round(dwell[row, column], 3).tolist()} N*m"
            )
    return dwell, joint_angles


def wait_for_pose(
    robot,
    controller: JointImpedanceController,
    pose: int,  # 0-indexed
) -> np.ndarray:  # (6,) rad, where the operator left the arm
    """Hold the arm under gravity compensation while the operator moves it by hand.

    stdin is polled rather than read, so the torque keeps streaming: the arm has
    no watchdog and latches its last command, and a blocking input() would leave
    a stale gravity torque computed for a pose the operator has since moved away
    from, which no longer holds the arm where it now is.
    """
    print(f"\npose {pose + 1}: backdrive the arm somewhere with room to sweep, then press Enter")
    period = 1.0 / CONTROL_FREQ_HZ
    joint_angles = np.array(robot.get_joint_angles().msg)
    while not select.select([sys.stdin], [], [], 0.0)[0]:
        start_time = time.monotonic()
        joint_angles = np.array(robot.get_joint_angles().msg)
        joint_velocities = read_joint_velocities(robot)
        apply_joint_torques(
            robot, controller.pin_model.nonlinear_effects(joint_angles, joint_velocities)
        )
        elapsed_time = time.monotonic() - start_time
        if elapsed_time < period:
            time.sleep(period - elapsed_time)
    sys.stdin.readline()
    print(f"pose {pose + 1}: q = {np.round(joint_angles, 4)} rad")
    return joint_angles


def sweep_joint(
    robot,
    controller: JointImpedanceController,
    feed_forward: FeedForward,
    where: tuple[int, int],  # pose, joint, both 0-indexed
    q_hold: np.ndarray,  # (6,) rad, the pose the other joints keep
    log: SweepLog,
) -> np.ndarray:  # (6,) rad, the q of the last cycle
    """Sweep one joint through every speed for SWEEP_SECONDS each, appending to log."""
    pose, joint = where
    limits = robot.get_config()["joint_limits"][f"joint{joint + 1}"]
    print(f"joint {joint + 1}: at {q_hold[joint]:.3f} rad, limits {np.round(limits, 3)} rad")
    joint_angles = q_hold
    for speed in SWEEP_SPEEDS:
        low, high = sweep_range(q_hold[joint], limits, speed)
        print(f"  {speed:5.3f} rad/s over {low:.3f} ... {high:.3f} rad for {SWEEP_SECONDS:.0f} s")
        joint_angles = sweep_at_speed(
            robot, controller, feed_forward, (pose, joint, speed, low, high), (q_hold, log)
        )
    return joint_angles


def sweep_at_speed(
    robot,
    controller: JointImpedanceController,
    feed_forward: FeedForward,
    sweep: tuple[int, int, float, float, float],  # pose, joint, rad/s, low rad, high rad
    state: tuple[np.ndarray, SweepLog],  # the pose the other joints keep, the log
) -> np.ndarray:  # (6,) rad, the q of the last cycle
    """Run one joint's triangle wave at one speed for SWEEP_SECONDS."""
    pose, joint, speed, low, high = sweep
    q_hold, log = state
    # A joint that starts within LIMIT_MARGIN of a limit lies outside its own
    # sweep, so the wave starts at the nearest end instead of jumping past it.
    q_wave_start = np.clip(q_hold[joint], low, high)
    period = 1.0 / CONTROL_FREQ_HZ
    joint_angles = q_hold
    joint_velocities = np.zeros_like(q_hold)

    t0 = time.monotonic()
    while time.monotonic() - t0 < SWEEP_SECONDS:
        start_time = time.monotonic()
        q_wave, qd_wave = triangle_wave(start_time - t0, low, high, speed, q_wave_start)

        q_des = q_hold.copy()
        qd_des = np.zeros_like(q_hold)
        q_des[joint], qd_des[joint] = q_wave, qd_wave
        # The triangle wave has zero acceleration away from its turnarounds, so
        # the feedforward here is friction only and does not depend on the
        # configuration it is asked for. The measured velocity is one cycle old,
        # 5 ms at 200 Hz.
        friction_ff = feed_forward.compute_torque(
            q_cur=q_hold,
            qd_cur=joint_velocities,
            qd_des=qd_des,
            qdd_des=np.zeros_like(q_hold),
        )

        joint_angles, joint_velocities, cmd_torque = run_control_cycle(
            robot, controller, q_des, qd_des, friction_ff
        )
        log.append((pose, joint, joint_angles, joint_velocities, cmd_torque, qd_wave))

        elapsed_time = time.monotonic() - start_time
        if elapsed_time < period:
            time.sleep(period - elapsed_time)
        else:
            print(f"warning: control loop overrun {elapsed_time:.3f}s > {period:.3f}s")
    return joint_angles


def run_control_cycle(
    robot,
    controller: JointImpedanceController,
    q_des: np.ndarray,  # (6,) rad
    qd_des: np.ndarray,  # (6,) rad/s
    extra_torque: np.ndarray,  # (6,) N*m, added to the impedance torque
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # q, qd, tau of this cycle
    """Read the state, send the joint impedance torque for this target plus extra_torque."""
    joint_angles = np.array(robot.get_joint_angles().msg)
    joint_velocities = read_joint_velocities(robot)

    cmd_torque = controller.compute_jnt_torque(
        q_des=q_des,
        v_des=qd_des,
        q_cur=joint_angles,
        v_cur=joint_velocities,
        base_orientation=R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix(),
    )
    cmd_torque = cmd_torque + extra_torque
    apply_joint_torques(robot, cmd_torque)
    return joint_angles, joint_velocities, cmd_torque


def log_to_arrays(log: SweepLog) -> dict[str, np.ndarray]:
    """The per-cycle log as (N,) and (N, 6) arrays, the form both the file and the fit want."""
    pose, joint, q, qd, tau, qd_des = (np.array([entry[i] for entry in log]) for i in range(6))
    return {"pose": pose, "joint": joint, "q": q, "qd": qd, "tau": tau, "qd_des": qd_des}


def friction_samples(
    pin_model,
    samples: dict[str, np.ndarray],
) -> np.ndarray:  # (N, 6) N*m
    """Friction of every sample, from the torque balance tau - nle(q, qd) - M(q) qdd.

    An identity rather than a steady-state assumption, so it holds while a joint
    is accelerating too. That is what makes the slow sweeps usable: the joints do
    not slide slowly on command, they stick and lurch, and masking the transients
    out -- as the commanded-speed fit this replaced had to -- throws away nearly
    all the low-speed data there is.
    """
    q, qd, tau = samples["q"], samples["qd"], samples["tau"]
    dofs = qd.shape[1]
    window = np.ones(QDD_SMOOTH_SAMPLES) / QDD_SMOOTH_SAMPLES
    qd_smooth = np.column_stack([np.convolve(qd[:, j], window, mode="same") for j in range(dofs)])
    qdd = np.gradient(qd_smooth, 1.0 / CONTROL_FREQ_HZ, axis=0)

    zero = np.zeros(dofs)
    friction = np.empty_like(tau)
    for i in range(len(q)):
        inertial = pin_model.inverse_dynamics(q[i], zero, qdd[i]) - pin_model.nonlinear_effects(
            q[i], zero
        )
        friction[i] = tau[i] - pin_model.nonlinear_effects(q[i], qd[i]) - inertial
    return friction


def curve_rows(
    friction: np.ndarray,  # (N,) N*m, one joint
    speed: np.ndarray,  # (N,) rad/s, signed, the same joint
) -> list[tuple[float, float, int, float, float]]:  # low, high, n, + direction, - direction
    """Median friction per measured-speed bin, the two directions kept apart.

    Folding the directions together hides a per-joint offset that is even in the
    velocity -- a model error, not friction -- and on joints 1 and 2 that offset
    is about 0.09 N*m against a friction of 0.4.
    """
    rows = []
    for low, high in pairwise(FRICTION_BIN_EDGES):
        in_bin = (np.abs(speed) >= low) & (np.abs(speed) < high)
        forward = in_bin & (speed > 0)
        backward = in_bin & (speed < 0)
        if min(forward.sum(), backward.sum()) < FRICTION_BIN_MIN_SAMPLES:
            continue
        rows.append(
            (
                low,
                high,
                int(in_bin.sum()),
                float(np.median(friction[forward])),
                float(-np.median(friction[backward])),
            )
        )
    return rows


def band_level(
    friction: np.ndarray,  # (N, 6) N*m
    samples: dict[str, np.ndarray],
    selection: np.ndarray,  # (N,) bool, which samples to use
) -> np.ndarray:  # (6,) N*m, nan where the band is too thin
    """Median friction of each joint over COULOMB_BAND, which is what FRICTION_COULOMB is.

    A median over a band of the MEASURED speed. A straight line through the
    commanded speeds is pulled about by whichever of them has the most samples,
    and that is how two earlier parameter sets came out wrong.
    """
    qd = samples["qd"]
    low, high = COULOMB_BAND
    level = np.full(qd.shape[1], np.nan)
    for joint in range(qd.shape[1]):
        speed = np.abs(qd[:, joint])
        in_band = selection & (samples["joint"] == joint) & (speed > low) & (speed < high)
        if in_band.sum() >= FRICTION_BIN_MIN_SAMPLES:
            level[joint] = np.median(friction[in_band, joint] * np.sign(qd[in_band, joint]))
    return level


def report_curve(friction: np.ndarray, samples: dict[str, np.ndarray]) -> None:
    """Print each joint's friction against measured speed, pooled over the poses."""
    qd = samples["qd"]
    for joint in range(qd.shape[1]):
        sel = samples["joint"] == joint
        print(f"\njoint {joint + 1}: friction (N*m) vs measured speed (rad/s)")
        print("   |v| band          n     +dir     -dir     mean")
        for low, high, count, forward, backward in curve_rows(friction[sel, joint], qd[sel, joint]):
            mean = 0.5 * (forward + backward)
            print(
                f"  {low:5.3f}-{high:5.3f} {count:7d}  {forward:7.3f}  {backward:7.3f}  {mean:7.3f}"
            )


def report_poses(
    friction: np.ndarray,
    samples: dict[str, np.ndarray],
    breakaway: np.ndarray,  # (poses, 6, 2, REPEATS) N*m
) -> None:
    """Print each joint's sliding and breakaway level at each pose, to show pose dependence."""
    poses = min(int(samples["pose"].max()) + 1, len(breakaway))
    levels = np.array([band_level(friction, samples, samples["pose"] == p) for p in range(poses)])
    ramps = np.nanmedian(np.abs(breakaway[:poses]), axis=(2, 3))
    print(f"\npose dependence: coulomb over {COULOMB_BAND} rad/s | breakaway median")
    for joint in range(samples["qd"].shape[1]):
        print(
            f"  joint {joint + 1}: {np.round(levels[:, joint], 3).tolist()}"
            f"  |  {np.round(ramps[:, joint], 3).tolist()}"
        )


def report_dwell(dwell: np.ndarray) -> None:  # (1, DWELL_JOINTS, 2, DWELL_TIMES) N*m
    """Print breakaway against time at rest, and whether it has a slope.

    A level that grows with dwell means FRICTION_STATIC as measured at
    BREAKAWAY_SETTLE_S is a floor, not the level a standing task actually meets.
    """
    if not dwell.size:
        return
    print("\nbreakaway after time at rest (N*m), pose 1")
    print("  joint dir " + "".join(f"{t:>8.0f} s" for t in DWELL_TIMES) + "   last/first")
    for row, joint in enumerate(DWELL_JOINTS):
        for column, sign in enumerate("-+"):
            ramps = dwell[0, row, column]
            ratio = ramps[-1] / ramps[0] if ramps[0] else float("nan")
            print(
                f"    {joint + 1}   {sign}  "
                + "".join(f"{x:9.3f} " for x in ramps)
                + f"   {ratio:6.2f}"
            )


def run_one_pose(
    robot,
    controller: JointImpedanceController,
    feed_forward: FeedForward,
    pose: int,  # 0-indexed
    q_hold: np.ndarray,  # (6,) rad
    collected: tuple[SweepLog, list[np.ndarray], list[np.ndarray]],  # log, ramps, dwell ramps
) -> np.ndarray:  # (6,) rad, q of the last cycle
    """Breakaway ramps for every joint, then a sweep of every joint, at one pose.

    The ramps are collected before the sweeps start rather than returned at the
    end, so a Ctrl-C during the sweeps still leaves this pose's ramps lined up
    with the samples it logged.
    """
    log, breakaway, dwell = collected
    ramps, joint_angles = measure_all_breakaway(robot, controller, q_hold)
    breakaway.append(ramps)
    # Once only, at the first pose: the question is how long the joint has been
    # still, not where it is, and repeating it per pose would double its minutes.
    if not pose:
        ramps, joint_angles = measure_dwell_breakaway(robot, controller, q_hold)
        dwell.append(ramps)
    for joint in range(robot.joint_nums):
        joint_angles = sweep_joint(robot, controller, feed_forward, (pose, joint), q_hold, log)
    return joint_angles


def main() -> None:
    robot = connect_arm()
    controller = make_controller(robot.joint_nums)
    feed_forward = FeedForward(urdf_path=str(URDF_PATH), dofs=robot.joint_nums)

    print(f"feeding the current friction forward: coulomb {FRICTION_COULOMB} N*m")
    print(
        f"{POSE_COUNT} poses; at each, {BREAKAWAY_REPEATS} breakaway ramps per joint per "
        f"direction at {BREAKAWAY_RAMP_RATE} N*m/s, then {SWEEP_SECONDS:.0f} s per joint at "
        f"each of {SWEEP_SPEEDS} rad/s. Ctrl-C to stop and fit what has been logged."
    )
    log: SweepLog = []
    breakaway: list[np.ndarray] = []
    dwell: list[np.ndarray] = []
    joint_angles = np.array(robot.get_joint_angles().msg)
    try:
        for pose in range(POSE_COUNT):
            q_hold = wait_for_pose(robot, controller, pose)
            joint_angles = run_one_pose(
                robot, controller, feed_forward, pose, q_hold, (log, breakaway, dwell)
            )
    except KeyboardInterrupt:
        print("\ninterrupted, holding current pose")
    finally:
        # Hold before fitting, so a failed fit cannot leave the arm unheld.
        hold_current_pose(robot, joint_angles)
        if log:
            save_and_fit(controller, log, np.array(breakaway), np.array(dwell))


def save_and_fit(
    controller: JointImpedanceController,
    log: SweepLog,
    breakaway: np.ndarray,  # (poses, 6, 2, REPEATS) N*m
    dwell: np.ndarray,  # (1, DWELL_JOINTS, 2, DWELL_TIMES) N*m
) -> None:
    samples = log_to_arrays(log)
    path = OUTPUT_DIR / f"friction_{datetime.now():%Y%m%d_%H%M%S}.npz"
    np.savez(
        path,
        breakaway=breakaway,
        dwell=dwell,
        dwell_times=np.array(DWELL_TIMES),
        dwell_joints=np.array(DWELL_JOINTS),
        **samples,
    )
    print(f"saved {len(log)} samples to {path}")

    friction = friction_samples(controller.pin_model, samples)
    report_curve(friction, samples)
    report_poses(friction, samples, breakaway)
    report_dwell(dwell)

    coulomb = band_level(friction, samples, np.ones(len(log), dtype=bool))
    static = np.nanmedian(np.abs(breakaway), axis=(0, 2, 3))
    print(f"\nin use now   coulomb {np.round(FRICTION_COULOMB, 4).tolist()}")
    print(f"             static  {np.round(FRICTION_STATIC, 4).tolist()}")
    print(f"change       coulomb {np.round(coulomb - FRICTION_COULOMB, 4).tolist()}")
    print(f"             static  {np.round(static - FRICTION_STATIC, 4).tolist()}")
    print("paste into controller/feed_forward.py:")
    print(f"FRICTION_COULOMB = np.array({np.round(coulomb, 4).tolist()})")
    print(f"FRICTION_STATIC = np.array({np.round(static, 4).tolist()})")


if __name__ == "__main__":
    main()
