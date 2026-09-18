#!/usr/bin/env python3
"""Identify the joint friction of the arm, for the FeedForward of play_joint_angles.py.

Holds the start pose under the joint impedance law of main_jnt_imp.py and sweeps
one joint at a time back and forth at a few constant speeds. At constant velocity
the impedance effort tau - nle is what the joint needs on top of the model, i.e.
its friction, so a fit of tau = f_viscous * qd + f_coulomb * sign(qd) over the
steady parts of the sweeps gives the two coefficients per joint.

The sweep runs with the friction of controller/feed_forward.py already fed
forward, so the joint tracks the commanded velocity instead of sticking while
the impedance spring winds up to the same torque. The fit is unaffected by that:
at steady velocity tau - nle is whatever the joint needs, however the command was
composed, so it returns the friction itself and not a residual. What improves is
the data. The printed change against the current values says how far off they were.

Before the sweeps it measures each joint's breakaway torque: the ramp torque at
which a standing joint first moves, in each direction. Nothing in the control law
uses it -- the feedforward is Coulomb only -- but it is the ceiling on how far the
Coulomb constant can sensibly be overestimated, since no joint ever needs more
than its breakaway torque.

The arm sweeps SWEEP_AMPLITUDE around the pose it starts in, so put it somewhere
with room to move every joint. Ctrl-C hands the arm to a position hold.

Run from workspace/src:  python identify_friction.py
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from controller.feed_forward import FRICTION_COULOMB, FeedForward, fit_friction
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
SWEEP_AMPLITUDE = 0.3  # rad, either side of the start pose
# The slowest speed is what the Coulomb constant is measured at: friction falls
# with speed, so a constant fitted over the fast sweeps under-compensates the
# slow motion the task actually does. It is also the hardest speed for the arm,
# and how well it holds it is the test of whether the compensation is enough.
SWEEP_SPEEDS = (0.05, 0.2, 0.4, 0.8)  # rad/s
# The band the Coulomb constant is taken over: the slowest speeds at which the
# joints genuinely slide rather than stick, which makes it a deliberate
# overestimate for faster motion.
SLIDING_BAND = (0.03, 0.10)  # rad/s
SWEEP_CYCLES = 2  # per speed
LIMIT_MARGIN = 0.1  # rad, kept free at each joint limit

# A joint moving this slowly is stuck rather than sliding, and the impedance
# effort it takes is the spring winding up, not friction.
FIT_MIN_SPEED = 0.02  # rad/s
# Samples this close to a turnaround are still accelerating.
FIT_TURNAROUND_S = 0.3

# Breakaway: the torque a joint needs before it moves at all, which the sweeps
# cannot see because it only exists while the joint is stopped. Ramped slowly so
# the joint stays quasi-static, and bounded well under the t_ff limits.
BREAKAWAY_RAMP_RATE = 0.25  # N*m/s
BREAKAWAY_MAX_TORQUE = 2.0  # N*m, above which the joint is called stuck
BREAKAWAY_MOTION = 0.01  # rad, counts as the joint having started to move
BREAKAWAY_SETTLE_S = 1.0  # s of plain hold, for the spring to take the joint back


def make_controller(dofs: int) -> JointImpedanceController:
    """JointImpedanceController with the gains validated in main_jnt_imp.py."""
    controller = JointImpedanceController(urdf_path=str(URDF_PATH), dofs=dofs)
    controller.set_jnt_params(
        b=np.array([0.5, 0.8, 0.8, 0.2, 0.2, 0.2]),
        k=np.array([10.0, 10.0, 10.0, 2.0, 1.0, 1.0]),
    )
    return controller


def sweep_range(q_start: float, joint_limits: tuple[float, float]) -> tuple[float, float]:
    """Travel of one joint: SWEEP_AMPLITUDE around q_start, inside its limits."""
    lower, upper = joint_limits
    return (
        max(q_start - SWEEP_AMPLITUDE, lower + LIMIT_MARGIN),
        min(q_start + SWEEP_AMPLITUDE, upper - LIMIT_MARGIN),
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
) -> tuple[np.ndarray, np.ndarray]:  # (6, 2) N*m for (-, +), q of the last cycle
    """Static friction of every joint in both directions, printed as it is measured."""
    breakaway = np.full((robot.joint_nums, 2), np.nan)
    joint_angles = q_hold
    q_target = q_hold.copy()
    for joint in range(robot.joint_nums):
        for column, direction in enumerate((-1.0, 1.0)):
            # Re-aim the spring at wherever the joint now is, before settling on
            # it. After a breakaway the joint coasts and stops where the spring
            # balances its static friction, and the spring is then wound up in
            # the opposite direction -- which would pay for most of the next
            # ramp and make the second direction read far too low.
            q_target[joint] = joint_angles[joint]
            joint_angles = hold_pose(robot, controller, q_target, BREAKAWAY_SETTLE_S)

            torque, joint_angles = measure_breakaway(robot, controller, joint, q_target, direction)
            breakaway[joint, column] = torque
            sign = "+" if direction > 0 else "-"
            print(f"joint {joint + 1} {sign}: breakaway {torque:.3f} N*m")
    return breakaway, joint_angles


def sweep_joint(
    robot,
    controller: JointImpedanceController,
    feed_forward: FeedForward,
    joint: int,  # 0-indexed
    q_hold: np.ndarray,  # (6,) rad, the pose the other joints keep
    log: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, float]],
) -> np.ndarray:  # (6,) rad, the q of the last cycle
    """Sweep one joint through every speed, logging (joint, q, qd, tau, qd_des)."""
    limits = robot.get_config()["joint_limits"][f"joint{joint + 1}"]
    low, high = sweep_range(q_hold[joint], limits)
    # A joint that starts within LIMIT_MARGIN of a limit lies outside its own
    # sweep, so the wave starts at the nearest end instead of jumping past it.
    q_wave_start = np.clip(q_hold[joint], low, high)
    print(
        f"joint {joint + 1}: at {q_hold[joint]:.3f} rad, sweeping "
        f"{low:.3f} ... {high:.3f} rad from {q_wave_start:.3f} rad "
        f"at {SWEEP_SPEEDS} rad/s, limits {np.round(limits, 3)} rad"
    )

    period = 1.0 / CONTROL_FREQ_HZ
    joint_angles = q_hold
    joint_velocities = np.zeros_like(q_hold)
    for speed in SWEEP_SPEEDS:
        t0 = time.monotonic()
        duration = SWEEP_CYCLES * 2.0 * (high - low) / speed
        while time.monotonic() - t0 < duration:
            start_time = time.monotonic()
            q_wave, qd_wave = triangle_wave(start_time - t0, low, high, speed, q_wave_start)

            q_des = q_hold.copy()
            qd_des = np.zeros_like(q_hold)
            q_des[joint], qd_des[joint] = q_wave, qd_wave
            # The triangle wave has zero acceleration away from its turnarounds,
            # which the fit drops anyway, so the feedforward here is friction
            # only and does not depend on the configuration it is asked for.
            # The measured velocity is one cycle old, 5 ms at 200 Hz.
            friction_ff = feed_forward.compute_torque(
                q_cur=q_hold,
                qd_cur=joint_velocities,
                qd_des=qd_des,
                qdd_des=np.zeros_like(q_hold),
            )

            joint_angles, joint_velocities, cmd_torque = run_control_cycle(
                robot, controller, q_des, qd_des, friction_ff
            )
            log.append((joint, joint_angles, joint_velocities, cmd_torque, qd_wave))

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


def log_to_arrays(
    log: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, float]],
) -> dict[str, np.ndarray]:
    """The per-cycle log as (N,) and (N, 6) arrays, the form both the file and the fit want."""
    joint, q, qd, tau, qd_des = (np.array([entry[i] for entry in log]) for i in range(5))
    return {"joint": joint, "q": q, "qd": qd, "tau": tau, "qd_des": qd_des}


def steady_velocity_mask(qd_des: np.ndarray) -> np.ndarray:  # (N,) bool
    """True where the target has been going the same way for FIT_TURNAROUND_S."""
    direction = np.sign(qd_des)
    change = np.flatnonzero(np.r_[True, direction[1:] != direction[:-1]])
    last_change = np.repeat(change, np.diff(np.r_[change, len(direction)]))
    return np.arange(len(direction)) - last_change >= FIT_TURNAROUND_S * CONTROL_FREQ_HZ


def fit_all_joints(
    controller: JointImpedanceController,
    samples: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:  # f_viscous (6,), f_coulomb (6,)
    """Friction of each joint from its own sweep samples, dropping the turnarounds."""
    q, qd, tau = samples["q"], samples["qd"], samples["tau"]
    # The impedance effort is tau - nle; what is left at constant velocity is friction.
    effort = np.array(
        [tau[i] - controller.pin_model.nonlinear_effects(q[i], qd[i]) for i in range(len(q))]
    )
    steady = steady_velocity_mask(samples["qd_des"])

    f_viscous = np.zeros(controller.dofs)
    f_coulomb = np.zeros(controller.dofs)
    for joint in range(controller.dofs):
        swept = steady & (samples["joint"] == joint) & (np.abs(qd[:, joint]) > FIT_MIN_SPEED)
        # The regressor is the commanded velocity, not the measured one: the
        # effort contains the impedance damping -b * qd, so measurement noise on
        # qd sits in both sides of the fit and drags the slope towards -b.
        f_viscous[joint], f_coulomb[joint] = fit_friction(
            samples["qd_des"][swept], effort[swept, joint]
        )
    return f_viscous, f_coulomb


def sliding_level(
    controller: JointImpedanceController,
    samples: dict[str, np.ndarray],
) -> np.ndarray:  # (6,) N*m
    """Median friction of each joint over SLIDING_BAND, which is what FRICTION_COULOMB is.

    A median rather than a fit: a straight line through every sweep speed is
    pulled about by whichever speeds happen to have the most samples, and that
    is how two earlier parameter sets came out wrong.
    """
    q, qd, tau = samples["q"], samples["qd"], samples["tau"]
    effort = np.array(
        [tau[i] - controller.pin_model.nonlinear_effects(q[i], qd[i]) for i in range(len(q))]
    )
    steady = steady_velocity_mask(samples["qd_des"])

    low, high = SLIDING_BAND
    level = np.zeros(controller.dofs)
    for joint in range(controller.dofs):
        speed = np.abs(qd[:, joint])
        in_band = steady & (samples["joint"] == joint) & (speed > low) & (speed < high)
        level[joint] = np.median(effort[in_band, joint] * np.sign(qd[in_band, joint]))
    return level


def main() -> None:
    robot = connect_arm()
    q_hold = np.array(robot.get_joint_angles().msg)
    controller = make_controller(robot.joint_nums)
    feed_forward = FeedForward(urdf_path=str(URDF_PATH), dofs=robot.joint_nums)

    print(f"q = {np.round(q_hold, 4)} rad")
    print(f"feeding the current friction forward: coulomb {FRICTION_COULOMB} N*m")
    print(f"ramping each joint to breakaway at {BREAKAWAY_RAMP_RATE} N*m/s, both directions")
    print(f"then sweeping every joint by ±{SWEEP_AMPLITUDE} rad around it; Ctrl-C to stop")
    log = []  # (joint, q, qd, tau, qd_des) per cycle
    joint_angles = q_hold
    breakaway = np.full((robot.joint_nums, 2), np.nan)
    try:
        breakaway, joint_angles = measure_all_breakaway(robot, controller, q_hold)
        for joint in range(robot.joint_nums):
            joint_angles = sweep_joint(robot, controller, feed_forward, joint, q_hold, log)
    except KeyboardInterrupt:
        print("\ninterrupted, holding current pose")
    finally:
        # Hold before fitting, so a failed fit cannot leave the arm unheld.
        hold_current_pose(robot, joint_angles)
        if log:
            save_and_fit(controller, log, breakaway)


def save_and_fit(
    controller: JointImpedanceController,
    log: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, float]],
    breakaway: np.ndarray,  # (6, 2) N*m
) -> None:
    samples = log_to_arrays(log)
    path = OUTPUT_DIR / f"friction_{datetime.now():%Y%m%d_%H%M%S}.npz"
    np.savez(path, breakaway=breakaway, **samples)
    print(f"saved {len(log)} samples to {path}")

    level = sliding_level(controller, samples)
    f_viscous, f_coulomb = fit_all_joints(controller, samples)
    print(
        f"breakaway, both directions  {np.round(np.nanmean(np.abs(breakaway), axis=1), 4).tolist()}"
    )
    print(f"straight-line fit, all speeds  coulomb {np.round(f_coulomb, 4).tolist()}")
    print(f"                               viscous {np.round(f_viscous, 4).tolist()}")
    print(f"in use now   {np.round(FRICTION_COULOMB, 4).tolist()}")
    print(f"change       {np.round(level - FRICTION_COULOMB, 4).tolist()}")
    print(f"paste into controller/feed_forward.py (median over {SLIDING_BAND} rad/s):")
    print(f"FRICTION_COULOMB = np.array({np.round(level, 4).tolist()})")


if __name__ == "__main__":
    main()
