"""The live ergodic exploration loop: one ergodic law, one impedance law, one arm.

Shared by the top-level scripts that run it -- run_ergodic_pipeline.py for a single
run and run_ergodic_trials.py for the E2T2 trialled experiment -- so the tuned
gains, the two loop rates and the ergodic hyperparameters exist exactly once.

The ergodic controller and the impedance controller are separate components and a
target pose is the whole interface between them. Control is fully online: the
measured pose is mapped into the ergodic controller's cube, the law advances it by
one u dt, and that cube position is mapped back out to a pose. Nothing is carried
between steps, so the statistics the law accumulates are the arm's own coverage.
Nothing else crosses either: a velocity expressed in cube units is not a physical
velocity, since the cube's orientation axes are half-angle quaternion logarithms
and its per-axis scales come from the demonstration's ranges.

The two rates are deliberately different, and they pull in opposite directions.
U_MAX * dt is the distance to the next setpoint, so a fast ergodic step would ask
for almost nothing -- at 200 Hz a few tenths of a millimetre, which the t_ff
quantisation rounds to zero torque. The torque loop meanwhile wants to be fast,
because the sampled closed loop is what bounds the gains. So the law sets a
setpoint at ERGODIC_FREQ_HZ and the torque goes out at CONTROL_FREQ_HZ.

Between two setpoints the commanded pose walks the straight line joining them, one
step per control cycle, arriving exactly as the next setpoint is computed. The arm
is therefore never asked to jump the whole distance at once. That also gives the
friction feedforward a desired velocity that is constant across the interval,
instead of one that faded as the arm closed on a standing target.

This module touches the arm: it streams torque through
execution.executor_helpers. The ergodic law itself is hardware-free and stays in
ergodic_controller/ergodic_controller.py.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from controller.feed_forward import FeedForward
from controller.task_imp_controller import CartesianImpedanceController, orientation_error
from direct_teaching.distribution.phase_projection import phase_context
from direct_teaching.distribution.pose_distribution import PoseDistribution
from direct_teaching.recorder.joint_angle_recorder import load_recording
from ergodic_controller.ergodic_controller import ErgodicController
from ergodic_controller.phase_ergodic_controller import PhaseErgodicController, PhaseTask
from execution.executor_helpers import apply_joint_torques, read_joint_velocities
from visualization.visualizer import LiveView

AGX_REFERENCE = Path(__file__).resolve().parents[1] / "agx_reference"
URDF_PATH = AGX_REFERENCE / "piper/piper/urdf/piper_description.urdf"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"

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
MAX_SPEED = 0.02  # m/s
# Joint 6's soft limit in the planar run, where nothing else holds rotation about
# the peg. A spring-damper on joint 6 alone, inside the last JOINT6_MARGIN before
# each URDF limit. At the limit the spring gives 1.5 N*m, about 9x joint 6's static
# friction (0.172 N*m), the most the friction feedforward ever pushes it with.
JOINT6_MARGIN = 0.3  # rad
JOINT6_K = 5.0  # N*m/rad
JOINT6_B = 0.2  # N*m*s/rad, main_jnt_imp.py's joint 6 damping

R_WORLD_BASE = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()


@dataclass
class Exploration:
    """The collaborators that live for a whole run, built before the arm is touched.

    The ergodic controller is here rather than rebuilt per run because its
    accumulated statistics are the point: run_ergodic_trials.py keeps one of these
    across every trial, so only the arm's pose is reset at a trial boundary.
    """

    controller: CartesianImpedanceController
    feed_forward: FeedForward
    distribution: PoseDistribution
    ergodic: ErgodicController | PhaseErgodicController
    live: LiveView
    # The law runs on x-y only; z and the peg's tilt stay at the start pose, and
    # rotation about the peg axis is left free.
    planar: bool = False


def make_controller(dofs: int) -> CartesianImpedanceController:
    """CartesianImpedanceController with the gains validated in main_tast_imp.py."""
    controller = CartesianImpedanceController(
        urdf_path=str(URDF_PATH), dofs=dofs, frame_name=TCP_FRAME_NAME
    )
    controller.set_joint_torque_weights(np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]))
    controller.set_cart_params(
        b=np.array([5.0, 5.0, 5.0, 0.2, 0.2, 0.1]),
        k=np.array([250.0, 250.0, 300.0, 3.0, 3.0, 1.0]),
    )
    return controller


def fit_distribution(
    controller: CartesianImpedanceController, t: np.ndarray, q: np.ndarray
) -> PoseDistribution:
    poses = [controller.pin_model.forward_kinematics(q_i, TCP_FRAME_NAME) for q_i in q]
    p = np.array([p_i for p_i, _ in poses])
    rotations = np.array([R_i for _, R_i in poses])
    return PoseDistribution(t, p, rotations, N_COMPONENTS)


def prepare_exploration(recording: Path) -> Exploration:
    """Everything offline, before the arm is touched -- the MeshCat scene included."""
    controller = make_controller(dofs=6)
    t_rec, q_rec = load_recording(recording)
    distribution = fit_distribution(controller, t_rec, q_rec)
    print(f"computing ergodic coefficients for {recording}")
    ergodic = ErgodicController(distribution.pdf, 6, ERGODIC_K, ERGODIC_N, U_MAX)
    live = LiveView(distribution, URDF_PATH, TCP_FRAME_NAME)
    return Exploration(
        controller=controller,
        feed_forward=FeedForward(urdf_path=str(URDF_PATH), dofs=6),
        distribution=distribution,
        ergodic=ergodic,
        live=live,
    )


def prepare_phase_exploration(master: Path, labelled: Path, beta: float = 1.0) -> Exploration:
    """prepare_exploration with the phase-conditioned law, before the arm is touched.

    labelled is a file from label_datapoint_phases.py. The datapoints' GMM is
    fitted only for its cube, so the gains, MAX_SPEED and the interpolator see the
    same coordinates as in run_ergodic_pipeline.py; the law itself uses the
    datapoints directly.
    """
    controller = make_controller(dofs=6)
    with np.load(labelled) as data:
        q, phi = data["q"], data["phi"]
    distribution = fit_distribution(controller, np.arange(len(q), dtype=float), q)
    context = phase_context(master, q, controller.pin_model)

    def to_cube(q_rows: np.ndarray) -> np.ndarray:  # (n, 6) rad -> (n, 6) cube
        poses = (controller.pin_model.forward_kinematics(q_i, TCP_FRAME_NAME) for q_i in q_rows)
        return np.array([distribution.pose_to_state(p_i, R_i) for p_i, R_i in poses])

    task = PhaseTask(
        X=to_cube(q),
        phi=phi,
        P_master=to_cube(context.q_master)[:, :2],
        phi_master=context.phi_master,
        # The cube scales each axis differently; back to metres, as the labelling projects.
        span=distribution.upper[:2] - distribution.lower[:2],
        sigma_f=context.sigma_f,
        axes=2,
    )
    print(f"phase: {len(q)} datapoints, sigma_f = lead = {context.sigma_f:.4f}, beta = {beta:g}")
    law = PhaseErgodicController(task, U_MAX, beta, ERGODIC_K)
    return Exploration(
        controller=controller,
        feed_forward=FeedForward(urdf_path=str(URDF_PATH), dofs=6),
        distribution=distribution,
        ergodic=law,
        live=LiveView(distribution, URDF_PATH, TCP_FRAME_NAME, phase_law=law),
        planar=True,
    )


def next_setpoint(
    exploration: Exploration,
    q: np.ndarray,  # (6,) rad, measured on this cycle
    dt: float,  # s, the ergodic step period -- not the control period
    x_cmd: np.ndarray,  # (6,) cube units, where the interpolator stands now
) -> np.ndarray:  # (6,) cube units
    """One ergodic step from wherever the arm is right now, as a cube state.

    Nothing is carried between calls. The measured pose goes in through
    pose_to_state and the law advances it by one u dt, so the statistics it
    accumulates are the arm's own coverage, and U_MAX * dt is exactly the distance
    to the next setpoint. It stays in the cube rather than becoming a pose here,
    because the interpolator between setpoints works in the cube too.

    A planar law steps x-y alone, and the rest is x_cmd's: the start pose's z and
    orientation, carried unchanged through every setpoint.
    """
    x = exploration.distribution.pose_to_state(
        *exploration.controller.pin_model.forward_kinematics(q, TCP_FRAME_NAME)
    )
    x = np.clip(x, 0.0, 1.0)
    if not exploration.planar:
        return exploration.ergodic.step(x, dt)
    return np.concatenate([exploration.ergodic.step(x[:2], dt), x_cmd[2:]])


def first_setpoints(
    exploration: Exploration,
    q: np.ndarray,  # (6,) rad, the pose the loop is about to start from
) -> tuple[np.ndarray, np.ndarray]:  # (6,) cube units each: where from, where to
    """Where the interpolator starts, and the first setpoint one ergodic step from it.

    Taken before the loop so its cycle 0 does not step the law a second time. This
    does accumulate one Phi(x) at the start pose, as every other step does -- which
    is why a trial's start pose counts as covered even if the arm leaves it at once.
    """
    p_tcp, R_tcp = exploration.controller.pin_model.forward_kinematics(q, TCP_FRAME_NAME)
    x_start = exploration.distribution.pose_to_state(p_tcp, R_tcp)
    return x_start, next_setpoint(exploration, q, 1.0 / ERGODIC_FREQ_HZ, x_start)


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


def print_start(
    exploration: Exploration,
    q: np.ndarray,  # (6,) rad
    setpoints: tuple[np.ndarray, np.ndarray],  # (6,) cube units each
) -> None:
    """The pose the loop starts from and the first setpoint it will walk toward."""
    p_tcp, _ = exploration.controller.pin_model.forward_kinematics(q, TCP_FRAME_NAME)
    p_first, _ = exploration.distribution.state_to_pose(setpoints[1])
    print(
        f"q = {np.round(q, 4)} rad, TCP {np.round(p_tcp, 4)} m\n"
        f"first setpoint {np.round(p_first, 4)} m, "
        f"{np.linalg.norm(p_first - p_tcp) * 1000:.2f} mm away, "
        f"approached over {round(CONTROL_FREQ_HZ / ERGODIC_FREQ_HZ)} cycles"
    )


def print_progress(
    exploration: Exploration,
    q: np.ndarray,  # (6,) rad
    p_target: np.ndarray,  # (3,) m
) -> None:
    """Every 20th ergodic step: ergodic_metric is a tensor-train norm, too slow for the loop.

    The phase law prints its phase state instead, which costs nothing: its metric
    needs the dense K^6 difference of target and statistic, 20 to 75 ms at a few
    thousand past states, several control cycles.
    """
    if exploration.ergodic.step_count % 20:
        return
    p_tcp, _ = exploration.controller.pin_model.forward_kinematics(q, TCP_FRAME_NAME)
    if isinstance(exploration.ergodic, PhaseErgodicController):
        phi, sigma_b, stall_over_T = exploration.ergodic.trace[-1]
        status = f"phi {phi:.3f}, sigma_b {sigma_b:.3f}, stall/T {stall_over_T:.2f}"
    else:
        status = f"ergodic metric {exploration.ergodic.ergodic_metric():.4f}"
    print(f"position error {np.linalg.norm(p_target - p_tcp):.5f} m, {status}")


def save_run(log: list[tuple], recording: Path, label: str = "") -> None:
    """Write the run for visualization.visualizer.show_ergodic_run.

    The recording is stored too: without it the distribution cannot be refitted,
    so nothing offline can say where the run sat relative to the cube.

    label is appended to the name. The timestamp resolves to a second, so a caller
    writing several runs in quick succession -- the trial sequence -- needs it to
    stop two of them landing on the same path.
    """
    t, q, p_target, R_target, tau = (np.array(col) for col in zip(*log, strict=True))
    path = OUTPUT_DIR / f"ergodic_run_{datetime.now():%Y%m%d_%H%M%S}{label}.npz"
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


def free_peg_rotation(
    R_cmd: np.ndarray,  # (3, 3), the commanded orientation
    R_tcp: np.ndarray,  # (3, 3), measured this cycle
) -> np.ndarray:  # (3, 3)
    """The measured orientation, tilted the shortest way onto R_cmd's peg axis.

    The peg axis is peg_tcp's z. The orientation error this leaves is perpendicular
    to the peg, so the impedance law holds the tilt and neither holds nor drives
    rotation about the peg. A fixed R_cmd with k_rz = 0 does not: the error is in
    world-aligned axes, so a rotation offset about the peg would leak into the tilt.
    """
    align, _ = R.align_vectors([R_cmd[:, 2]], [R_tcp[:, 2]])
    return align.as_matrix() @ R_tcp


def commanded_pose(
    exploration: Exploration,
    x_cmd: np.ndarray,  # (6,) cube units, where the interpolator stands
    q: np.ndarray,  # (6,) rad, measured this cycle
) -> tuple[np.ndarray, np.ndarray]:  # (3,) m, (3, 3)
    """The pose x_cmd stands for; a planar run leaves rotation about the peg free."""
    p_cmd, R_cmd = exploration.distribution.state_to_pose(x_cmd)
    if not exploration.planar:
        return p_cmd, R_cmd
    _, R_tcp = exploration.controller.pin_model.forward_kinematics(q, TCP_FRAME_NAME)
    return p_cmd, free_peg_rotation(R_cmd, R_tcp)


def joint6_limit_torque(
    q6: float,  # rad
    qd6: float,  # rad/s
    lower: float,  # rad, joint 6's limits
    upper: float,  # rad
) -> float:  # N*m
    """Zero inside the soft band; beyond it, a spring back to the band's edge plus damping.

    Joint 6's axis is the peg axis, so this torque turns only the rotation the
    planar run leaves free, and the Cartesian law contributes none about it.
    """
    excess = q6 - np.clip(q6, lower + JOINT6_MARGIN, upper - JOINT6_MARGIN)
    if excess == 0.0:
        return 0.0
    return -JOINT6_K * excess - JOINT6_B * qd6


def run_control_cycle(
    robot,
    exploration: Exploration,
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
    cmd_torque = exploration.controller.compute_cartesian_torque(
        desired_pos=target[0],
        desired_ori=target[1],
        q_cur=q,
        v_cur=joint_velocities,
        base_orientation=R_WORLD_BASE,
    )
    cmd_torque = cmd_torque + exploration.feed_forward.compute_torque(
        q_cur=q,
        qd_cur=joint_velocities,
        qd_des=desired_joint_velocity(exploration.controller, q, twist),
        qdd_des=np.zeros(robot.joint_nums),
    )
    if exploration.planar:
        model = exploration.controller.pin_model.robot.model
        cmd_torque[5] += joint6_limit_torque(
            q[5], joint_velocities[5], model.lowerPositionLimit[5], model.upperPositionLimit[5]
        )
    apply_joint_torques(robot, cmd_torque)
    return cmd_torque


def explore(
    robot,
    exploration: Exploration,
    setpoints: tuple[np.ndarray, np.ndarray],  # (6,) cube units each: where from, where to
    log: list,  # appended in place, so the caller still has it after an interrupt
    stop: Callable[[np.ndarray], bool] | None = None,
) -> float:  # s, how long the loop ran
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

    stop is passed the joint angles this cycle measured and is polled once per
    cycle, after the torque has gone out -- never before, so whatever it does
    cannot delay a torque. None means the loop only ends on KeyboardInterrupt or an
    exception, which is a single open-ended run; the trialled experiment passes a
    predicate that fires when the peg reaches the hole.
    """
    distribution = exploration.distribution
    x_cmd, x_next = setpoints
    period = 1.0 / CONTROL_FREQ_HZ
    ergodic_period = 1.0 / ERGODIC_FREQ_HZ
    cycles_per_step = round(CONTROL_FREQ_HZ / ERGODIC_FREQ_HZ)
    step, twist = plan_interval(distribution, x_cmd, x_next, ergodic_period, cycles_per_step)
    t0 = time.monotonic()
    # while True rather than a for over itertools.count(), so the only way out is a
    # return, an interrupt or an exception, and there is no unreachable tail.
    cycle = -1
    while True:
        cycle += 1
        start_time = time.monotonic()
        joint_angles = np.array(robot.get_joint_angles().msg)
        # Not at cycle 0: the caller has already taken that step.
        if cycle and cycle % cycles_per_step == 0:
            x_next = next_setpoint(exploration, joint_angles, ergodic_period, x_cmd)
            step, twist = plan_interval(
                distribution, x_cmd, x_next, ergodic_period, cycles_per_step
            )
        x_cmd = x_cmd + step
        target = commanded_pose(exploration, x_cmd, joint_angles)
        cmd_torque = run_control_cycle(robot, exploration, joint_angles, target, twist)
        log.append((start_time - t0, joint_angles, target[0], target[1], cmd_torque))
        # Only on a step: step_count holds still between them, so a progress line
        # gated on it alone would repeat on every cycle of the interval.
        if cycle % cycles_per_step == 0:
            print_progress(exploration, joint_angles, target[0])
            exploration.live.update(joint_angles, *target)
        if stop is not None and stop(joint_angles):
            return time.monotonic() - t0

        elapsed_time = time.monotonic() - start_time
        if elapsed_time < period:
            time.sleep(period - elapsed_time)
        else:
            print(f"warning: control loop overrun {elapsed_time:.3f}s > {period:.3f}s")
