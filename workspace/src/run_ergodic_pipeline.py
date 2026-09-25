#!/usr/bin/env python3
"""Ergodic exploration of a recording's pose distribution on the live arm.

Fits the time-weighted PoseDistribution to a recording, computes the E2T2
coefficients, and lets the ergodic controller move a Cartesian target that the
main_tast_imp.py impedance law tracks.

The two are separate components and a target pose is the whole interface between
them. Control is fully online: the measured pose is mapped into the ergodic
controller's cube, the law advances it by one u dt, and that cube position is
mapped back out to a pose. Nothing is carried between steps, so the statistics
the law accumulates are the arm's own coverage. Nothing else crosses either: a
velocity expressed in cube units is not a physical velocity, since the cube's
orientation axes are half-angle quaternion logarithms and its per-axis scales
come from the demonstration's ranges.

The two rates are deliberately different, and they pull in opposite directions.
U_MAX * dt is the distance to the next setpoint, so a fast ergodic step would ask
for almost nothing -- at 200 Hz a few tenths of a millimetre, which the t_ff
quantisation rounds to zero torque. The torque loop meanwhile wants to be fast,
because the sampled closed loop is what bounds the gains. So the law sets a
setpoint at ERGODIC_FREQ_HZ and the torque goes out at CONTROL_FREQ_HZ.

Between two setpoints the commanded pose walks the straight line joining them,
one step per control cycle, arriving exactly as the next setpoint is computed.
The arm is therefore never asked to jump the whole distance at once. That also
gives the friction feedforward a desired velocity that is constant across the
interval, instead of one that faded as the arm closed on a standing target.

Ctrl-C hands the arm to a position hold.

Run from workspace/src:  python run_ergodic_pipeline.py recording.npz
"""

import argparse
import itertools
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from controller.feed_forward import FeedForward
from controller.task_imp_controller import CartesianImpedanceController, orientation_error
from direct_teaching.distribution.pose_distribution import PoseDistribution
from direct_teaching.recorder.joint_angle_recorder import load_recording
from ergodic_controller.ergodic_controller import ErgodicController
from execution.executor_helpers import (
    apply_joint_torques,
    connect_arm,
    hold_current_pose,
    read_joint_velocities,
)
from visualization.visualizer import LiveView

AGX_REFERENCE = Path(__file__).resolve().parent / "agx_reference"
URDF_PATH = AGX_REFERENCE / "piper/piper/urdf/piper_description.urdf"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

CONTROL_FREQ_HZ = 100.0
ERGODIC_FREQ_HZ = 20.0
TCP_FRAME_NAME = "peg_tcp"

# As visualization/visualizer.py: 8 components as in the E2T2 paper, Fourier
# modes and quadrature points per dimension from the E2T2 notebook.
N_COMPONENTS = 8
ERGODIC_K = 10
ERGODIC_N = 10
U_MAX = 3.0  # cube units per second
# Ceiling on how fast the commanded pose may travel, whatever U_MAX and the
# demonstration's span between them would ask for. It also bounds the first
# interval, where the arm can start outside the cube with the setpoint far away.
MAX_SPEED = 0.05  # m/s

R_WORLD_BASE = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()


def make_controller(dofs: int) -> CartesianImpedanceController:
    """CartesianImpedanceController with the gains validated in main_tast_imp.py."""
    controller = CartesianImpedanceController(
        urdf_path=str(URDF_PATH), dofs=dofs, frame_name=TCP_FRAME_NAME
    )
    controller.set_joint_torque_weights(np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]))
    controller.set_cart_params(
        b=np.array([4.0, 4.0, 4.0, 0.1, 0.1, 0.1]),
        k=np.array([200.0, 200.0, 200.0, 1.0, 1.0, 1.0]),
    )
    return controller


def fit_distribution(
    controller: CartesianImpedanceController, t: np.ndarray, q: np.ndarray
) -> PoseDistribution:
    poses = [controller.pin_model.forward_kinematics(q_i, TCP_FRAME_NAME) for q_i in q]
    p = np.array([p_i for p_i, _ in poses])
    rotations = np.array([R_i for _, R_i in poses])
    return PoseDistribution(t, p, rotations, N_COMPONENTS)


def prepare_exploration(
    recording: Path,
) -> tuple[CartesianImpedanceController, PoseDistribution, ErgodicController, LiveView]:
    """Everything offline, before the arm is touched -- the MeshCat scene included."""
    controller = make_controller(dofs=6)
    t_rec, q_rec = load_recording(recording)
    distribution = fit_distribution(controller, t_rec, q_rec)
    print(f"computing ergodic coefficients for {recording}")
    ergodic = ErgodicController(distribution.pdf, 6, ERGODIC_K, ERGODIC_N, U_MAX)
    live = LiveView(distribution, URDF_PATH, TCP_FRAME_NAME)
    return controller, distribution, ergodic, live


def next_setpoint(
    controller: CartesianImpedanceController,
    distribution: PoseDistribution,
    ergodic: ErgodicController,
    q: np.ndarray,  # (6,) rad, measured on this cycle
    dt: float,  # s, the ergodic step period -- not the control period
) -> np.ndarray:  # (6,) cube units
    """One ergodic step from wherever the arm is right now, as a cube state.

    Nothing is carried between calls. The measured pose goes in through
    pose_to_state and the law advances it by one u dt, so the statistics it
    accumulates are the arm's own coverage, and U_MAX * dt is exactly the distance
    to the next setpoint. It stays in the cube rather than becoming a pose here,
    because the interpolator between setpoints works in the cube too.
    """
    x = distribution.pose_to_state(*controller.pin_model.forward_kinematics(q, TCP_FRAME_NAME))
    return ergodic.step(np.clip(x, 0.0, 1.0), dt)


def plan_interval(
    distribution: PoseDistribution,
    x_from: np.ndarray,  # (6,) cube units, where the interpolator stands now
    x_next: np.ndarray,  # (6,) cube units, the setpoint to walk toward
    dt: float,  # s, the ergodic step period
    cycles: int,  # control cycles in one ergodic interval
) -> tuple[np.ndarray, np.ndarray]:  # (6,) cube units per cycle, (6,) m/s and rad/s
    """The per-cycle interpolation step, and the task velocity it travels at.

    The straight line from x_from to x_next would be walked in one ergodic period.
    MAX_SPEED caps how fast the commanded pose may travel along it, and when the
    cap binds the interval ends short of the setpoint: the next interval then
    starts from wherever the interpolator actually reached, so the commanded pose
    never jumps. That does not accumulate, because every setpoint is one ergodic
    step from the measured pose -- capped, the whole thing proceeds at MAX_SPEED
    instead of falling behind without bound.

    The speed is measured between the poses the endpoints map to, never between
    the cube states: a velocity in cube units is not a physical one, since the
    orientation axes are half-angle quaternion logarithms.

    Scaling the whole step keeps the path's direction, so the rotation slows by the
    same factor as the translation. Only the linear speed is capped, so a segment
    that is almost pure rotation is bounded by the setpoint geometry rather than by
    MAX_SPEED.
    """
    p_from, R_from = distribution.state_to_pose(x_from)
    p_next, R_next = distribution.state_to_pose(x_next)
    twist = np.concatenate([p_next - p_from, orientation_error(R_next, R_from)]) / dt
    speed = float(np.linalg.norm(twist[:3]))
    scale = min(1.0, MAX_SPEED / speed) if speed > 0.0 else 1.0
    return scale * (x_next - x_from) / cycles, scale * twist


def print_progress(
    controller: CartesianImpedanceController,
    ergodic: ErgodicController,
    q: np.ndarray,  # (6,) rad
    p_target: np.ndarray,  # (3,) m
) -> None:
    """Every 20th ergodic step: ergodic_metric is a tensor-train norm, too slow for the loop."""
    if ergodic.step_count % 20:
        return
    p_tcp, _ = controller.pin_model.forward_kinematics(q, TCP_FRAME_NAME)
    print(
        f"position error {np.linalg.norm(p_target - p_tcp):.5f} m, "
        f"ergodic metric {ergodic.ergodic_metric():.4f}"
    )


def save_run(log: list[tuple], recording: Path) -> None:
    """Write the run for visualization.visualizer.show_ergodic_run.

    The recording is stored too: without it the distribution cannot be refitted,
    so nothing offline can say where the run sat relative to the cube.
    """
    t, q, p_target, R_target, tau = (np.array(col) for col in zip(*log, strict=True))
    path = OUTPUT_DIR / f"ergodic_run_{datetime.now():%Y%m%d_%H%M%S}.npz"
    np.savez(
        path, t=t, q=q, p_target=p_target, R_target=R_target, tau=tau, recording=str(recording)
    )
    print(f"saved {len(t)} cycles to {path}")


def desired_joint_velocity(
    controller: CartesianImpedanceController,
    q: np.ndarray,  # (6,) rad
    twist: np.ndarray,  # (6,) m/s and rad/s, the interpolator's velocity
) -> np.ndarray:  # (6,) rad/s
    """The joint velocity that walks the line between setpoints, least-norm through J.

    FeedForward uses it only for the direction of the friction it applies, so
    nothing is compensated while the interpolator stands still and the arm stays
    backdrivable. Near a singular configuration the pseudo-inverse makes this
    large, but only the sign reaches the torque, which stays bounded by f_static.
    """
    return np.linalg.pinv(controller.pin_model.jacobian(q, TCP_FRAME_NAME)) @ twist


# The loop's collaborators, named rather than bundled behind a partial or a tuple.
def explore(  # noqa: PLR0913, PLR0917
    robot,
    controller: CartesianImpedanceController,
    feed_forward: FeedForward,
    distribution: PoseDistribution,
    ergodic: ErgodicController,
    live: LiveView,
    setpoints: tuple[np.ndarray, np.ndarray],  # (6,) cube units each: where from, where to
    log: list,  # appended in place, so the caller still has it after an interrupt
) -> None:
    """Stream torque at CONTROL_FREQ_HZ, re-stepping the ergodic law at ERGODIC_FREQ_HZ.

    The law sets a setpoint every ergodic period, U_MAX * dt away in the cube.
    Between two of them the commanded pose walks the straight line joining them,
    one step per control cycle, so the arm is never asked to jump the whole
    distance at once. x_cmd is where that walk has reached; it is what is
    commanded, and it is where the next interval starts from, so capping the speed
    in plan_interval simply ends an interval short rather than causing a jump.

    One q read per cycle serves both laws. It used to be two, a control period
    apart, which online control cannot afford: the setpoint is one ergodic step
    from the measured pose, so a stale q puts the arm's own motion into its own
    setpoint.

    The log is passed in and appended in place so the caller still holds every
    cycle recorded before an exception, not only before a clean interrupt.
    """
    x_cmd, x_next = setpoints
    period = 1.0 / CONTROL_FREQ_HZ
    ergodic_period = 1.0 / ERGODIC_FREQ_HZ
    cycles_per_step = round(CONTROL_FREQ_HZ / ERGODIC_FREQ_HZ)
    step, twist = plan_interval(distribution, x_cmd, x_next, ergodic_period, cycles_per_step)
    t0 = time.monotonic()
    for cycle in itertools.count():
        start_time = time.monotonic()
        joint_angles = np.array(robot.get_joint_angles().msg)
        # Not at cycle 0: the caller has already taken that step.
        if cycle and cycle % cycles_per_step == 0:
            x_next = next_setpoint(controller, distribution, ergodic, joint_angles, ergodic_period)
            step, twist = plan_interval(
                distribution, x_cmd, x_next, ergodic_period, cycles_per_step
            )
        x_cmd = x_cmd + step
        target = distribution.state_to_pose(x_cmd)
        cmd_torque = run_control_cycle(robot, controller, feed_forward, joint_angles, target, twist)
        log.append((start_time - t0, joint_angles, target[0], target[1], cmd_torque))
        # Only on a step: step_count holds still between them, so a progress line
        # gated on it alone would repeat on every cycle of the interval.
        if cycle % cycles_per_step == 0:
            print_progress(controller, ergodic, joint_angles, target[0])
            live.update(joint_angles, *target)

        elapsed_time = time.monotonic() - start_time
        if elapsed_time < period:
            time.sleep(period - elapsed_time)
        else:
            print(f"warning: control loop overrun {elapsed_time:.3f}s > {period:.3f}s")


def main(recording: Path) -> None:
    controller, distribution, ergodic, live = prepare_exploration(recording)

    feed_forward = FeedForward(urdf_path=str(URDF_PATH), dofs=6)

    robot = connect_arm()
    joint_angles = np.array(robot.get_joint_angles().msg)
    p_tcp, R_tcp = controller.pin_model.forward_kinematics(joint_angles, TCP_FRAME_NAME)
    x_start = distribution.pose_to_state(p_tcp, R_tcp)
    x_first = next_setpoint(controller, distribution, ergodic, joint_angles, 1.0 / ERGODIC_FREQ_HZ)
    p_first, _ = distribution.state_to_pose(x_first)
    cycles_per_step = round(CONTROL_FREQ_HZ / ERGODIC_FREQ_HZ)
    print(
        f"q = {np.round(joint_angles, 4)} rad, TCP {np.round(p_tcp, 4)} m\n"
        f"first setpoint {np.round(p_first, 4)} m, "
        f"{np.linalg.norm(p_first - p_tcp) * 1000:.2f} mm away, "
        f"approached over {cycles_per_step} cycles\n"
        f"exploring; Ctrl-C to stop"
    )
    # (t s, q (6,) rad, p_target (3,) m, R_target (3, 3), tau (6,) N*m) per cycle
    log = []
    try:
        explore(
            robot, controller, feed_forward, distribution, ergodic, live, (x_start, x_first), log
        )
    except KeyboardInterrupt:
        print("\ninterrupted, holding current pose")
    finally:
        # Hold before writing, so a failed write cannot leave the arm unheld.
        hold_current_pose(robot, log[-1][1] if log else joint_angles)
        if log:
            save_run(log, recording)
    if ergodic.step_count:
        print(f"ergodic metric: {ergodic.ergodic_metric():.4f}")


def run_control_cycle(
    robot,
    controller: CartesianImpedanceController,
    feed_forward: FeedForward,
    q: np.ndarray,  # (6,) rad, measured this cycle
    target: tuple[np.ndarray, np.ndarray],  # (3,) m, (3, 3), this cycle's interpolated pose
    twist: np.ndarray,  # (6,) m/s and rad/s, constant across the ergodic interval
) -> np.ndarray:  # (6,) N*m, the torque sent
    """Send the torque tracking the target pose from the measured q, plus feedforward.

    qdd_des is zero, so FeedForward's inertia term cancels exactly and what it adds
    is friction alone. The interpolator moves at a constant velocity across each
    interval, so there is no second derivative of it to use.
    """
    joint_velocities = read_joint_velocities(robot)
    cmd_torque = controller.compute_cartesian_torque(
        desired_pos=target[0],
        desired_ori=target[1],
        q_cur=q,
        v_cur=joint_velocities,
        base_orientation=R_WORLD_BASE,
    )
    cmd_torque = cmd_torque + feed_forward.compute_torque(
        q_cur=q,
        qd_cur=joint_velocities,
        qd_des=desired_joint_velocity(controller, q, twist),
        qdd_des=np.zeros(robot.joint_nums),
    )
    apply_joint_torques(robot, cmd_torque)
    return cmd_torque


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Mandatory: the arm explores whatever recording this is, so never pick one implicitly.
    parser.add_argument("recording", type=Path, help="joint_angles_*.npz")
    args = parser.parse_args()
    main(args.recording)
