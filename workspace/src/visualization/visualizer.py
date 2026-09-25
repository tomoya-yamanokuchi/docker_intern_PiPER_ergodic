"""Offline views of recordings: matplotlib in the browser on :8988, MeshCat on :7000.

Hardware-free: python visualization/visualizer.py [recording.npz] shows that
recording, or the newest one in workspace/output/ if none is given.
"""

import argparse
import threading
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import meshcat
import numpy as np

from controller.feed_forward import STRIBECK_VELOCITY
from core.agx_pinocchio import AgxPinocchio, helper
from direct_teaching.distribution.pose_distribution import PoseDistribution
from direct_teaching.recorder.joint_angle_recorder import load_recording
from ergodic_controller.ergodic_controller import ErgodicController
from kinematics.kinematic_solver import KinematicSolver
from simulation.meshcat_scene import (
    animate_robot,
    draw_axes,
    draw_joint_sweeps,
    draw_pdf_cloud,
    draw_position_distribution,
    draw_tcp_paths,
    show_robot,
)

matplotlib.use("WebAgg")
plt.rcParams["webagg.port"] = 8988
plt.rcParams["webagg.open_in_browser"] = False

SRC = Path(__file__).resolve().parents[1]
URDF_PATH = SRC / "agx_reference/piper/piper/urdf/piper_description.urdf"
OUTPUT_DIR = SRC.parent / "output"

PDF_POINTS = 5000
# LiveView frames between one trail point and the next; the trail is resent whole.
TRAIL_DECIMATION = 5
# How long LiveView's drawing thread waits when the caller has handed it nothing.
IDLE_POLL_PERIOD = 0.005  # s
POSITION_DIMS = [0, 1, 2]
ORIENTATION_DIMS = [3, 4, 5]
LABELS = [f"$X_{i + 1}$ [{unit}]" for i, unit in enumerate(["m"] * 3 + ["rad"] * 3)]
GRID_POINTS = 100

# Fourier modes and quadrature points per dimension, from the E2T2 notebook.
ERGODIC_K = 5
ERGODIC_N = 10

# ErgodicController's c: across this band at each face the ergodic command is
# blended out in favour of a velocity toward the cube centre.
CENTRE_BAND = 0.05
# A PoseDistribution built with margin 0.1 puts its data in exactly this range,
# because the cube is 1.2 times the data's own width on every axis.
DATA_RANGE = (1 / 12, 11 / 12)

# t_ff quantisation step and clamp, pyAgxArm PiperFW.DEFAULT. A commanded torque
# below half a step encodes to the same value as no command at all.
TORQUE_STEP = np.array([0.251, 0.251, 0.251, 0.051, 0.051, 0.051])  # (6,) N*m
TORQUE_LIMIT = np.array([32.0, 32.0, 32.0, 6.506, 6.506, 6.506])  # (6,) N*m
# Below this a joint counts as standing: feed_forward.py's compensation band,
# so the same speed that module treats as standstill is used here.
STILL_SPEED = STRIBECK_VELOCITY  # (6,) rad/s


def _impedance_torque(
    pin_model: AgxPinocchio,
    q: np.ndarray,  # (N, 6) rad
    qd: np.ndarray,  # (N, 6) rad/s
    tau: np.ndarray,  # (N, 6) N*m, as sent
) -> np.ndarray:  # (N, 6) N*m
    """The part of the sent torque that carries the command, not the arm's weight.

    What goes out is dominated by the nonlinear effects holding the arm up, which
    say nothing about tracking. What is left after subtracting them is what has to
    clear half a t_ff quantisation step to reach the joint at all.
    """
    return np.array(
        [
            tau_i - pin_model.nonlinear_effects(q_i, qd_i)
            for tau_i, q_i, qd_i in zip(tau, q, qd, strict=True)
        ]
    )


def _print_stall_diagnosis(commanding: np.ndarray, moving: np.ndarray) -> None:
    """Why each joint is or is not following, as a share of cycles.

    The three outcomes are exclusive and exhaustive, so each joint's row sums to
    100, and which column is large is the diagnosis: 'rounds away' means the
    impedance torque never reached the joint, 'stalled' means it did and the joint
    did not move anyway, which is friction.
    """
    print(f"\n{'joint':>6s} {'rounds away':>12s} {'stalled':>9s} {'moving':>8s}")
    for j in range(commanding.shape[1]):
        c, m = commanding[:, j], moving[:, j]
        print(
            f"{j + 1:>6d} {(~c).mean() * 100:>11.1f}% {(c & ~m).mean() * 100:>8.1f}%"
            f" {m.mean() * 100:>7.1f}%"
        )


def _show_tracking(
    pin_model: AgxPinocchio,
    axes,  # three axes: joint speed, then impedance torque for joints 1-3 and 4-6
    t: np.ndarray,  # (N,) s
    q: np.ndarray,  # (N, 6) rad
    tau: np.ndarray,  # (N, 6) N*m, as sent
) -> None:
    """Why the arm is or is not moving toward the setpoint, read across two signals.

    Online control recomputes the target from the measured pose every step, so the
    position error is the commanded lead whatever the arm does and cannot answer
    this. Joint speed and impedance torque can, read together:

        torque inside the quantisation band  -> the command never reached the joint
        torque outside it but the joint still -> friction is holding it
        joint moving                          -> it is following

    qd is a central difference of the logged q, so it lags what the loop read by
    half a cycle. STILL_SPEED is feed_forward.py's Stribeck band, the speed below
    which that module already treats a joint as standing.
    """
    qd = np.gradient(q, t, axis=0)
    contribution = _impedance_torque(pin_model, q, qd, tau)
    commanding = np.abs(contribution) > TORQUE_STEP / 2.0
    moving = np.abs(qd) > STILL_SPEED
    print(f"peak |tau| sent {np.round(np.abs(tau).max(axis=0), 3)} of {TORQUE_LIMIT} N*m")
    _print_stall_diagnosis(commanding, moving)

    for j in range(6):
        axes[0].plot(t, np.abs(qd[:, j]), linewidth=0.8, label=f"joint {j + 1}")
    axes[0].axhspan(0.0, STILL_SPEED.max(), color="#d9534f", alpha=0.12, zorder=0)
    axes[0].set_ylabel("joint speed [rad/s]")
    axes[0].set_yscale("log")
    for ax, joints in ((axes[1], (0, 1, 2)), (axes[2], (3, 4, 5))):
        ax.axhspan(
            -TORQUE_STEP[joints[0]] / 2,
            TORQUE_STEP[joints[0]] / 2,
            color="#d9534f",
            alpha=0.12,
            zorder=0,
        )
        for j in joints:
            ax.plot(t, contribution[:, j], linewidth=0.8, label=f"joint {j + 1}")
        ax.set_ylabel(f"impedance $\\tau$ [N m]\njoints {joints[0] + 1}-{joints[-1] + 1}")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.grid(color="0.9", linewidth=0.5)


def _sample_marginal(
    distribution: PoseDistribution, dims: list[int], n: int, rng: np.random.Generator
) -> np.ndarray:  # (n, len(dims)) cube coordinates
    """Draw from the marginal GMM; a grid misses components narrower than its spacing."""
    counts = rng.multinomial(n, distribution.priors)
    means = distribution.means[:, dims]
    covariances = distribution.covariances[:, dims][:, :, dims]
    draws = zip(means, covariances, counts, strict=True)
    return np.concatenate([rng.multivariate_normal(m, c, k) for m, c, k in draws])


def _cube_to_physical(distribution: PoseDistribution, X: np.ndarray, dims: list[int]) -> np.ndarray:
    lower, upper = distribution.lower[dims], distribution.upper[dims]
    return lower + X * (upper - lower)


def _plot_marginal_contour(
    ax: plt.Axes,
    distribution: PoseDistribution,
    dims: list[int],  # two cube axes
    samples: np.ndarray,  # (N, 2) physical units
) -> None:
    """Contours of the 2D marginal pdf, with a trajectory on top, as in the E2T2 notebook."""
    axis = np.linspace(0.0, 1.0, GRID_POINTS)
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    density = distribution.marginal_pdf(grid, dims).reshape(GRID_POINTS, GRID_POINTS)
    x, y = _cube_to_physical(distribution, np.column_stack([axis, axis]), dims).T
    ax.contourf(x, y, density.T, alpha=0.5)
    ax.plot(*samples.T, "-k", linewidth=0.5)
    ax.set_xlabel(LABELS[dims[0]])
    ax.set_ylabel(LABELS[dims[1]])


def _cube_axis_labels(distribution: PoseDistribution) -> list[str]:
    """Per-axis label carrying the physical width of the cube on that axis."""
    span = distribution.upper - distribution.lower
    # The orientation axes hold half-angle quaternion logs, so the rotation they
    # span is twice the axis width.
    widths = [(w, "m") for w in span[:3]] + [(2 * w, "rad") for w in span[3:]]
    return [f"$X_{i + 1}$  ({w:.4f} {unit})" for i, (w, unit) in enumerate(widths)]


def _report_wall_contact(distribution: PoseDistribution, X: np.ndarray) -> None:  # (N, 6)
    """How much of the trajectory sits where the ergodic command is being blended out."""
    span = distribution.upper - distribution.lower
    in_band = (X < CENTRE_BAND) | (X > 1.0 - CENTRE_BAND)
    print(f"cube width: {np.round(span[:3], 4)} m, {np.round(2 * span[3:], 4)} rad")
    print(
        f"fraction of time in the pull-to-centre band, per axis: "
        f"{np.round(in_band.mean(axis=0), 3)}"
    )
    print(f"in the band on at least one axis: {in_band.any(axis=1).mean():.3f} of the time")


def _plot_cube_axes(t: np.ndarray, X: np.ndarray, distribution: PoseDistribution) -> None:
    """One panel per cube axis on fixed [0, 1], position left, orientation right."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, layout="constrained")
    fig.suptitle("Cube state per axis; shaded = pull-to-centre band, dashed = demonstrated range")
    labels = _cube_axis_labels(distribution)
    for ax, label, x_axis in zip(axes.T.ravel(), labels, X.T, strict=True):
        ax.axhspan(0.0, CENTRE_BAND, color="#d62728", alpha=0.15)
        ax.axhspan(1.0 - CENTRE_BAND, 1.0, color="#d62728", alpha=0.15)
        for edge in DATA_RANGE:
            ax.axhline(edge, color="0.5", linestyle="--", linewidth=0.8)
        ax.plot(t, x_axis, color="#1f5f99", linewidth=0.8)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(color="0.9", linewidth=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("t [s]")
    axes[0, 0].set_xlim(t[0], t[-1])


def _homogeneous(p: np.ndarray, R: np.ndarray) -> np.ndarray:  # (3,) m, (3, 3) -> (4, 4)
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = p
    return transform


class LiveView:
    """The scene of a run as it happens: the distribution, the arm, the newest setpoint.

    Everything static is drawn in __init__, which is why it is built before the arm
    is connected: the pdf cloud of the distribution being explored, and the URDF
    meshes at the zero pose, which jump to the measured pose on the first update.

    A frame draws the arm at the measured q, its TCP triad, and a longer triad at
    the commanded pose, so a tracking error is the gap between the two triads. The
    trails behind them -- measured in black, commanded in red -- are what shows
    coverage of the cloud while the run is still going.

    **No drawing happens in the caller's thread.** update() only stores the newest
    state; a daemon thread draws it. A MeshCat message is a zmq round trip to the
    server process, and one costs about 10 ms when the calls are a control loop's
    50 ms apart -- a full control period, though the same call costs under 1 ms
    when they come back to back and the server is hot. So the sends go to a thread,
    which is safe to do because pyzmq drops the GIL while it blocks. That is also
    why the view owns its own AgxPinocchio: robot.data is mutable scratch space that
    the controller writes on every control cycle.

    A trail is resent whole, so it is appended to and redrawn only every
    TRAIL_DECIMATION-th frame; coverage does not need the full rate to be legible.
    """

    def __init__(
        self,
        distribution: PoseDistribution,
        urdf_path: Path = URDF_PATH,
        frame_name: str = "peg_tcp",
    ):
        self.pin_model = AgxPinocchio(str(urdf_path))
        self.frame_name = frame_name
        self.measured: list[np.ndarray] = []
        self.commanded: list[np.ndarray] = []
        self.frames = 0
        # The newest state update() has handed over, or None once it is drawn.
        self.pending: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.viewer = meshcat.Visualizer()
        print(f"meshcat: {self.viewer.url()}")
        rng = np.random.default_rng(0)
        points = _sample_marginal(distribution, POSITION_DIMS, PDF_POINTS, rng)
        log_density = np.log10(distribution.marginal_pdf(points, POSITION_DIMS))
        draw_pdf_cloud(
            self.viewer, _cube_to_physical(distribution, points, POSITION_DIMS), log_density
        )
        self.robot_view = show_robot(self.viewer, self.pin_model.robot, np.zeros(6), frame_name)
        draw_axes(self.viewer["target"], length=0.08, radius=0.002)
        threading.Thread(target=self._draw_pending, daemon=True).start()

    def update(
        self,
        q: np.ndarray,  # (6,) rad, measured
        p_target: np.ndarray,  # (3,) m, commanded this cycle
        R_target: np.ndarray,  # (3, 3), commanded this cycle
    ) -> None:
        """Hand the newest state to the drawing thread. Stores one tuple, draws nothing."""
        self.pending = (q, p_target, R_target)

    def _draw_pending(self) -> None:
        """Draw whatever update() last handed over, as fast as the sends allow.

        Only the newest state is ever wanted, so a state overwritten before this
        thread picks it up is simply skipped, as is one written into the gap between
        the read and the clear below. A dropped frame of a viewer is invisible.
        """
        while True:
            pending, self.pending = self.pending, None
            if pending is None:
                time.sleep(IDLE_POLL_PERIOD)
                continue
            q, p_target, R_target = pending
            p, rotation = self.pin_model.forward_kinematics(q, self.frame_name)
            self.robot_view.display(q)
            self.viewer["tcp"].set_transform(_homogeneous(p, rotation))
            self.viewer["target"].set_transform(_homogeneous(p_target, R_target))
            self.frames += 1
            if self.frames % TRAIL_DECIMATION == 0:
                self.measured.append(p)
                self.commanded.append(p_target)
                draw_tcp_paths(self.viewer, np.array(self.measured), np.array(self.commanded))


class Visualizer:
    def __init__(self, urdf_path: Path = URDF_PATH, frame_name: str = "peg_tcp"):
        self.urdf_path = urdf_path
        self.pin_model = AgxPinocchio(str(urdf_path))
        self.frame_name = frame_name

    def show_joint_pose(
        self, q: np.ndarray, sweep_amplitude: float | None = None
    ) -> meshcat.Visualizer:  # q (6,) rad, sweep rad
        """One static pose of the arm in MeshCat, with the TCP axes on the frame.

        With a sweep amplitude, the TCP path of each joint swept on its own is
        drawn on top, one colour per joint.
        """
        q = np.asarray(q, dtype=float)
        p, rotation = self.pin_model.forward_kinematics(q, self.frame_name)
        print(f"q = {np.round(q, 4)} rad")
        print(f"{self.frame_name} at {np.round(p, 4)} m, z axis {np.round(rotation[:, 2], 3)}")
        viewer = meshcat.Visualizer()
        print(f"meshcat: {viewer.url()}")
        show_robot(viewer, self.pin_model.robot, q, self.frame_name)
        if sweep_amplitude is not None:
            draw_joint_sweeps(viewer, self.pin_model.robot, q, sweep_amplitude, self.frame_name)
        return viewer

    def show_wall_contact(
        self,
        recording_path: Path,
        n_components: int,
        duration: float = 30.0,  # s
        dt: float = 0.01,  # s
        u_max: float = 0.5,  # cube units per second
    ) -> None:
        """Why a 6-D run stalls: the cube state against the pull-to-centre band.

        Simulates the ergodic trajectory offline, so it needs no arm. The
        orientation axes span only a few degrees of real rotation, so the state
        reaches their band within seconds and the ergodic command is replaced
        there by a pull toward the cube centre.
        """
        t_rec, q_rec = load_recording(recording_path)
        distribution, X_samples = self._fit_distribution(t_rec, q_rec, n_components)
        controller = ErgodicController(distribution.pdf, 6, ERGODIC_K, ERGODIC_N, u_max)
        X = [X_samples[len(X_samples) // 2]]
        for _ in range(int(duration / dt)):
            X.append(controller.step(X[-1], dt))
        X = np.array(X)
        print(f"ergodic metric: {controller.ergodic_metric():.4f}")
        _report_wall_contact(distribution, X)
        _plot_cube_axes(dt * np.arange(len(X)), X, distribution)
        print("matplotlib: http://127.0.0.1:8988")
        plt.show()

    def show_ergodic_run(self, run_path: Path) -> None:
        """Commanded against measured TCP for a run on the arm, from run_ergodic_pipeline.py.

        Control is fully online, so the commanded pose is one ergodic step from the
        measured one and the two paths nearly overlie; the error panels are where
        the gap is legible. That gap should sit near the commanded lead, u dt
        mapped out of the cube, and a much larger one means the arm is not
        following.
        """
        run = np.load(run_path)
        t, q, p_target, R_target = run["t"], run["q"], run["p_target"], run["R_target"]
        poses = [self.pin_model.forward_kinematics(q_i, self.frame_name) for q_i in q]
        p = np.array([p_i for p_i, _ in poses])
        position_error = np.linalg.norm(p_target - p, axis=1)
        orientation_error = np.array(
            [
                np.linalg.norm(helper.orientation_error_rotmat(R_t, R_i))
                for R_t, (_, R_i) in zip(R_target, poses, strict=True)
            ]
        )
        print(
            f"{len(t)} cycles over {t[-1]:.1f} s, TCP travelled "
            f"{np.linalg.norm(np.diff(p, axis=0), axis=1).sum():.4f} m"
        )
        print(
            f"position error    median {np.median(position_error):.5f} m, "
            f"max {position_error.max():.5f} m"
        )
        print(
            f"orientation error median {np.median(orientation_error):.5f} rad, "
            f"max {orientation_error.max():.5f} rad"
        )

        viewer = meshcat.Visualizer()
        print(f"meshcat: {viewer.url()}")
        draw_tcp_paths(viewer, p, p_target)
        animate_robot(viewer, self.pin_model.robot, t, q, self.frame_name)

        # Runs recorded before the torque was logged still show their error panels.
        rows = 5 if "tau" in run.files else 2
        _, axes = plt.subplots(rows, 1, figsize=(10, 3 * rows), sharex=True, layout="constrained")
        for ax, error, label in (
            (axes[0], position_error, "position error [m]"),
            (axes[1], orientation_error, "orientation error [rad]"),
        ):
            ax.plot(t, error, color="#1f5f99", linewidth=0.8)
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0.0)
            ax.grid(color="0.9", linewidth=0.5)
        if rows == 5:
            _show_tracking(self.pin_model, axes[2:], t, q, run["tau"])
        axes[-1].set_xlabel("t [s]")
        axes[-1].set_xlim(t[0], t[-1])
        print("matplotlib: http://127.0.0.1:8988")
        plt.show()

    def show_pose_distribution(self, recording_path: Path, n_components: int) -> None:
        """Fit a PoseDistribution to one recording and show its marginals with the recording on top."""
        t, q = load_recording(recording_path)
        distribution, X_samples = self._fit_distribution(t, q, n_components)
        self._show(distribution, X_samples, t, q)

    def show_ergodic_trajectory(
        self,
        recording_path: Path,
        n_components: int,
        duration: float = 30.0,  # s
        dt: float = 0.01,  # s
        u_max: float = 0.5,  # cube units per second
    ) -> None:
        """Generate an ergodic trajectory over a recording's distribution offline and show it.

        Starts at the recording's first pose. A state IK cannot reach keeps the
        previous joint angles.
        """
        t_rec, q_rec = load_recording(recording_path)
        distribution, X_samples = self._fit_distribution(t_rec, q_rec, n_components)
        controller = ErgodicController(distribution.pdf, 6, ERGODIC_K, ERGODIC_N, u_max)
        X = [X_samples[0]]
        for _ in range(int(duration / dt)):
            X.append(controller.step(X[-1], dt))
        print(f"ergodic metric: {controller.ergodic_metric():.4f}")

        solver = KinematicSolver(str(self.urdf_path))
        q, unreachable = [q_rec[0]], 0
        for x in X[1:]:
            q_i = solver.inverse_kinematics(*distribution.state_to_pose(x), q_init=q[-1])
            unreachable += q_i is None
            q.append(q[-1] if q_i is None else q_i)
        print(f"unreachable by IK: {unreachable} of {len(X) - 1} states")
        self._show(distribution, np.array(X), dt * np.arange(len(X)), np.array(q))

    def _fit_distribution(
        self, t: np.ndarray, q: np.ndarray, n_components: int
    ) -> tuple[PoseDistribution, np.ndarray]:  # distribution, (N, 6) cube states
        poses = [self.pin_model.forward_kinematics(q_i, self.frame_name) for q_i in q]
        p = np.array([p_i for p_i, _ in poses])
        R = np.array([R_i for _, R_i in poses])
        distribution = PoseDistribution(t, p, R, n_components)
        X = np.array([distribution.pose_to_state(p_i, R_i) for p_i, R_i in poses])
        return distribution, X

    def _show(
        self,
        distribution: PoseDistribution,
        X: np.ndarray,  # (N, 6) cube states of the trajectory
        t: np.ndarray,  # (N,) s
        q: np.ndarray,  # (N, 6) rad
    ) -> None:
        """Marginal contours in matplotlib; pdf, trajectory and robot animation in MeshCat."""
        samples_physical = _cube_to_physical(distribution, X, POSITION_DIMS + ORIENTATION_DIMS)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
        fig.suptitle("Fitted distribution for multiple dimensions")
        for row, space_dims in enumerate((POSITION_DIMS, ORIENTATION_DIMS)):
            for col, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
                dims = [space_dims[i], space_dims[j]]
                ax = axes[row, col]
                _plot_marginal_contour(ax, distribution, dims, samples_physical[:, dims])

        rng = np.random.default_rng(0)
        points = _sample_marginal(distribution, POSITION_DIMS, PDF_POINTS, rng)
        log_density = np.log10(distribution.marginal_pdf(points, POSITION_DIMS))
        viewer = meshcat.Visualizer()
        print(f"meshcat: {viewer.url()}")
        points_physical = _cube_to_physical(distribution, points, POSITION_DIMS)
        draw_position_distribution(
            viewer, points_physical, log_density, samples_physical[:, POSITION_DIMS]
        )
        animate_robot(viewer, self.pin_model.robot, t, q, self.frame_name)

        print("matplotlib: http://127.0.0.1:8988")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "recording", nargs="?", type=Path, help="joint_angles_*.npz; default: newest in output/"
    )
    args = parser.parse_args()
    recording = args.recording or sorted(OUTPUT_DIR.glob("joint_angles_*.npz"))[-1]
    if recording.name.startswith("ergodic_run_"):
        Visualizer().show_ergodic_run(recording)
        raise SystemExit
    # 8 components, as the E2T2 paper selected for its demonstrations.
    # Visualizer().show_pose_distribution(recording, n_components=8)
    Visualizer().show_ergodic_trajectory(recording, 8, 60)
    # Visualizer().show_wall_contact(recording, 8, 60)
