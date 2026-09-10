#!/usr/bin/env python3
# -*-coding:utf8-*-

import time
from dataclasses import dataclass, field

from impedance_control.mit import compose_mit_targets, send_mit
from impedance_control.piper import get_q
from impedance_control.telemetry import send_sample

# ds/dt peaks at 1.5 halfway through a smoothstep, so a move sized by its
# average speed would overshoot the requested limit by half. Travel time is
# stretched by this factor to make max_speed a real ceiling.
SMOOTHSTEP_PEAK = 1.5


@dataclass
class MotionProfile:
    """
    How a streamed MIT move is executed, as opposed to where it goes.

    Every move in a run shares one of these -- the gains, the feedforward model
    and the command rate do not change between waypoints, only the target does.
    Passing them as one object also keeps move_to() and mit2can_park() inside
    the six-argument limit the quality gate enforces.

    tau_ff_fn(q, qdot, qddot) -> {joint: tau} supplies the feedforward torque.
    It is evaluated at every step, never interpolated between endpoints: the
    torque is a strongly non-linear function of configuration, and interpolating
    it is wrong by several Nm even on an ordinary move.

    telemetry is an optional socket from open_telemetry(); with one, every step
    is streamed to the visualization application as a single non-blocking sendto.
    """

    tuning: dict = field(default_factory=dict)
    tau_ff_fn: object = None
    max_speed: float = 0.3  # rad/s, of the joint with furthest to go
    rate: float = 100.0  # Hz, command rate of the streamed trajectory
    telemetry: object = None


def smoothstep(t):
    """
    Interpolation fraction and its first two derivatives at t in [0, 1].

    s = 3t^2 - 2t^3, so velocity is zero at both ends. A linear ramp would step
    the velocity at the start and the stop, which kd then fights.
    """
    return 3.0 * t**2 - 2.0 * t**3, 6.0 * t - 6.0 * t**2, 6.0 - 12.0 * t


def plan_travel(q_start, q_target, max_speed, rate):
    """
    Per-joint deltas, travel time and step count for one smoothstep move.

    Returns None when there is nothing to travel, so the caller can say so and
    skip the whole trajectory rather than divide by a zero duration.
    """
    joints = sorted(q_target)

    dq = {joint: q_target[joint] - q_start[joint] for joint in joints}
    travel = max(abs(value) for value in dq.values())

    if travel < 1e-6:
        return None

    duration = SMOOTHSTEP_PEAK * travel / max_speed

    return dq, travel, duration, max(1, round(duration * rate))


def step_references(q_start, dq, fraction, duration, tau_ff_fn):
    """
    Commanded angle, velocity and feedforward torque at one point of the move.

    fraction runs 0 to 1 over the trajectory; duration [s] converts the
    smoothstep derivatives into real rad/s and rad/s^2.
    """
    s, ds, dds = smoothstep(fraction)

    q = {joint: q_start[joint] + s * delta for joint, delta in dq.items()}
    qdot = {joint: delta * ds / duration for joint, delta in dq.items()}
    qddot = {joint: delta * dds / duration**2 for joint, delta in dq.items()}

    return q, qdot, tau_ff_fn(q, qdot, qddot) if tau_ff_fn is not None else None


def move_to(piper, interface, q_target, profile, report_every=1.0, dry_run=False):
    """
    Travel from the current pose to q_target as a streamed MIT trajectory.

    q_target is a {joint: q} dict in rad; profile is a MotionProfile carrying
    the gains, the feedforward model, the speed and the command rate. Travel
    time comes from profile.max_speed applied to the joint that has furthest to
    go, so a short move is quick and a long one is not violent.

    Every report_every seconds it prints the commanded angles next to the
    measured ones, because the gap between them is what the tuning is about.
    With profile.telemetry the same pair is streamed every step instead, for
    the visualization application to draw; that is one non-blocking sendto and nothing
    else, so a run with no viewer listening is unaffected.

    With dry_run nothing is sent and the whole command stream is printed
    instead, so a trajectory can be inspected with the arm unpowered.
    """
    joints = sorted(q_target)
    q_start = get_q(piper, joints)

    plan = plan_travel(q_start, q_target, profile.max_speed, profile.rate)

    if plan is None:
        print("INFO: Already at the target, nothing to travel.")
        return

    dq, travel, duration, steps = plan

    print(
        f"INFO: Moving {travel:.3f} rad over {duration:.1f}s "
        f"in {steps} steps at {profile.rate:.0f} Hz"
        + (" [dry run: nothing sent, arm not read]" if dry_run else "")
    )

    reported = 0.0
    deadline = time.perf_counter()

    for step in range(steps + 1):
        elapsed = step * duration / steps

        q, qdot, tau_ff = step_references(q_start, dq, step / steps, duration, profile.tau_ff_fn)
        targets = compose_mit_targets(q, profile.tuning, qdot_ref=qdot, tau_ff=tau_ff)

        if dry_run:
            print(f"  t={elapsed:5.2f}s q={_row(q)} qdot={_row(qdot)} tau_ff={_row(tau_ff)}")
            continue

        send_mit(interface, targets, quiet=True)

        if profile.telemetry is not None:
            send_sample(profile.telemetry, elapsed, q, get_q(piper, joints), qdot, tau_ff)

        if elapsed - reported >= report_every:
            reported = elapsed
            report_tracking(f"t+{elapsed:4.1f}s", q, get_q(piper, joints), joints)

        # Sleep to the next deadline rather than a fixed interval, so the work
        # in the step does not stretch the trajectory.
        deadline += 1.0 / profile.rate
        time.sleep(max(0.0, deadline - time.perf_counter()))

    if dry_run:
        print("INFO: Travel finished at:", q_target)
    else:
        report_tracking("arrived", q_target, get_q(piper, joints), joints)


def mit2can_park(piper, interface, q_park, profile, hold_time=5.0, move_spd_rate_ctrl=20):
    """
    Graceful shutdown: travel to q_park under MIT, then hand over to CAN MOVE J
    holding the zero pose, so the arm keeps it after the program exits.
    The handover target is a hardcoded JointCtrl(0, ...), so q_park is expected
    to be the zero pose.

    The travel is a streamed move_to() trajectory rather than a single MIT
    setpoint: a one-shot command leaves the whole distance to the position
    error, which at a soft kp the arm crosses slowly and never quite finishes,
    so it was still short of the pose when MOVE J took over and closed the
    remainder in one swing.

    Follows the vendor's own MIT parking path, handle_go_zero_service() in
    double_PiPER/src/piper/scripts/piper_ctrl_single_node.py:

        MotionCtrl_2(0x01, 0x01, 50, 0xAD)   # CAN ctrl, MOVE J, MIT byte kept
        JointCtrl(0, 0, 0, 0, 0, 0)

    Coming out of MIT the mode frame keeps is_mit_mode = 0xAD and only switches
    move_mode to MOVE J. The arm never passes through STANDBY, so the drivers are
    never dropped and there is no torque gap -- unlike reset_from_mit(), which
    follows piper_ctrl_reset.py through ctrl_mode = 0x00.

    Sending 0x00 in that byte instead is what leaves the arm limp, and that is
    what plain move_j does: __high_follow_mode defaults to 0. So the two frames
    are sent directly rather than through move_j.
    """
    # Travel to the park pose compliantly, before the stiff position loop
    # takes over.
    move_to(piper, interface, q_park, profile)

    time.sleep(hold_time)

    # The gap between this and the zero pose is the swing MOVE J is about to
    # close in one go, at move_spd_rate_ctrl.
    print("INFO: Handing over from:", get_q(piper))

    interface.ModeCtrl(0x01, 0x01, move_spd_rate_ctrl, 0xAD)
    interface.JointCtrl(0, 0, 0, 0, 0, 0)

    time.sleep(hold_time)

    print("INFO: Parked in CAN mode at:", get_q(piper))
    print("INFO: Drivers enabled:", interface.GetArmEnableStatus())


def _row(values):
    """
    One line of per-joint numbers.
    """
    if values is None:
        return "-"

    return "[" + " ".join(f"{values[joint]:7.3f}" for joint in sorted(values)) + "]"


def report_tracking(label, commanded, measured, joints):
    """
    Print what was commanded next to what the arm actually did.

    The commanded row is the point of this: measured angles alone cannot show
    whether the arm is tracking, and at a soft kp with no feedforward the gap is
    the whole story -- it is the gravity sag, joint by joint.
    """
    error = {joint: measured[joint] - commanded[joint] for joint in joints}
    worst = max(joints, key=lambda joint: abs(error[joint]))

    print(f"INFO: {label}")
    print(f"        cmd {_row(commanded)}")
    print(f"        act {_row(measured)}")
    print(f"        err {_row(error)}   worst {error[worst]:+.3f} rad on j{worst}")
