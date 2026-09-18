"""Offline views of recordings: matplotlib in the browser on :8988, MeshCat on :7000.

Hardware-free: python visualization/visualizer.py [recording.npz] shows that
recording, or the newest one in workspace/output/ if none is given.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import meshcat
import numpy as np

from core.agx_pinocchio import AgxPinocchio
from direct_teaching.distribution.pose_distribution import PoseDistribution
from direct_teaching.recorder.joint_angle_recorder import load_recording
from ergodic_controller.ergodic_controller import ErgodicController
from kinematics.kinematic_solver import KinematicSolver
from simulation.meshcat_scene import (
    animate_robot,
    draw_joint_sweeps,
    draw_position_distribution,
    show_robot,
)

matplotlib.use("WebAgg")
plt.rcParams["webagg.port"] = 8988
plt.rcParams["webagg.open_in_browser"] = False

SRC = Path(__file__).resolve().parents[1]
URDF_PATH = SRC / "agx_reference/piper/piper/urdf/piper_description.urdf"
OUTPUT_DIR = SRC.parent / "output"

PDF_POINTS = 5000
POSITION_DIMS = [0, 1, 2]
ORIENTATION_DIMS = [3, 4, 5]
LABELS = [f"$X_{i + 1}$ [{unit}]" for i, unit in enumerate(["m"] * 3 + ["rad"] * 3)]
GRID_POINTS = 100

# Fourier modes and quadrature points per dimension, from the E2T2 notebook.
ERGODIC_K = 5
ERGODIC_N = 10


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
    # 8 components, as the E2T2 paper selected for its demonstrations.
    # Visualizer().show_pose_distribution(recording, n_components=8)
    Visualizer().show_ergodic_trajectory(recording, 8, 60)
