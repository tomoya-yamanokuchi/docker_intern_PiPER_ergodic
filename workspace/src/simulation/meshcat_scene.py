"""MeshCat drawing in the robot's world frame, served on :7000."""

import math

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import pinocchio as pin
from matplotlib import colormaps
from meshcat.animation import Animation
from pinocchio.visualize import MeshcatVisualizer

PDF_OPACITY = 0.3


def _draw_axes(viewer: meshcat.Visualizer, length: float, radius: float) -> None:  # m
    """RGB axes as cylinders; WebGL ignores line width, so a triad cannot be made thicker."""
    # A meshcat cylinder lies along its own y axis, centred on the origin.
    axes = (
        ("x", 0xFF0000, tf.rotation_matrix(-np.pi / 2, [0, 0, 1]), [length / 2, 0, 0]),
        ("y", 0x00FF00, np.eye(4), [0, length / 2, 0]),
        ("z", 0x0000FF, tf.rotation_matrix(np.pi / 2, [1, 0, 0]), [0, 0, length / 2]),
    )
    for name, color, rotation, offset in axes:
        viewer[name].set_object(g.Cylinder(length, radius), g.MeshLambertMaterial(color=color))
        viewer[name].set_transform(tf.translation_matrix(offset) @ rotation)


def draw_position_distribution(
    viewer: meshcat.Visualizer,
    pdf_p: np.ndarray,  # (M, 3) m, drawn from the pdf
    log_density: np.ndarray,  # (M,)
    p_samples: np.ndarray,  # (N, 3) m
) -> None:
    """Draws from the pdf coloured by log density, and the recorded flange positions in black."""
    span = np.ptp(log_density)
    color = colormaps["viridis"]((log_density - log_density.min()) / span)[:, :3]
    viewer["distribution/position"].set_object(
        g.PointCloud(pdf_p.T.astype(np.float32), color.T.astype(np.float32), size=0.005)
    )
    # PointsMaterial takes no opacity; the viewer's "color" property sets it, and a white
    # material colour leaves the per-point colours unchanged.
    viewer["distribution/position"].set_property("color", [1.0, 1.0, 1.0, PDF_OPACITY])
    viewer["distribution/samples"].set_object(
        g.PointCloud(
            p_samples.T.astype(np.float32), np.zeros_like(p_samples.T, np.float32), size=0.001
        )
    )


def draw_tcp_paths(
    viewer: meshcat.Visualizer,
    p_measured: np.ndarray,  # (N, 3) m
    p_target: np.ndarray,  # (N, 3) m
) -> None:
    """Where the TCP went, in black, against where it was commanded to go, in red."""
    for name, p, color in (
        ("run/measured", p_measured, [0.0, 0.0, 0.0]),
        ("run/target", p_target, [1.0, 0.0, 0.0]),
    ):
        colors = np.tile(np.array(color, np.float32), (len(p), 1))
        viewer[name].set_object(g.PointCloud(p.T.astype(np.float32), colors.T, size=0.002))


def show_robot(
    viewer: meshcat.Visualizer,
    robot: pin.RobotWrapper,
    q: np.ndarray,  # (6,) rad
    tcp_frame: str,
) -> MeshcatVisualizer:
    """Draws the URDF visual meshes in one pose, with the TCP axes on the frame."""
    robot_view = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    robot_view.initViewer(viewer=viewer)
    robot_view.loadViewerModel()
    robot_view.display(q)
    tcp_data = robot.model.createData()
    pin.framesForwardKinematics(robot.model, tcp_data, q)
    _draw_axes(viewer["tcp"], length=0.05, radius=0.002)
    viewer["tcp"].set_transform(tcp_data.oMf[robot.model.getFrameId(tcp_frame)].homogeneous)
    return robot_view


def draw_joint_sweeps(
    viewer: meshcat.Visualizer,
    robot: pin.RobotWrapper,
    q: np.ndarray,  # (6,) rad
    amplitude: float,  # rad, either side of q
    tcp_frame: str,
) -> None:
    """The TCP path traced by sweeping each joint on its own, the others held at q.

    The travel is not clipped to the joint limits; the caller checks that.
    """
    data = robot.model.createData()
    tcp_id = robot.model.getFrameId(tcp_frame)
    colors = colormaps["turbo"](np.linspace(0.05, 0.95, q.shape[0]))[:, :3]
    for joint, color in enumerate(colors):
        path = []
        for angle in q[joint] + np.linspace(-amplitude, amplitude, 101):
            q_swept = q.copy()
            q_swept[joint] = angle
            pin.framesForwardKinematics(robot.model, data, q_swept)
            path.append(data.oMf[tcp_id].translation.copy())
        points = np.array(path).T.astype(np.float32)
        viewer[f"sweep/joint{joint + 1}"].set_object(
            g.PointCloud(
                points, np.tile(color[:, None], points.shape[1]).astype(np.float32), size=0.004
            )
        )


def animate_robot(
    viewer: meshcat.Visualizer,
    robot: pin.RobotWrapper,
    t: np.ndarray,  # (N,) s
    q: np.ndarray,  # (N, 6) rad
    tcp_frame: str,
) -> None:
    """Plays the URDF visual meshes and the TCP axes along a recording, keyed at the recorded t."""
    robot_view = show_robot(viewer, robot, q[0], tcp_frame)
    tcp_id = robot.model.getFrameId(tcp_frame)
    tcp_data = robot.model.createData()
    # Keyframes in seconds, so gaps from loop overruns play back at their real duration.
    animation = Animation(default_framerate=1)
    for t_i, q_i in zip(t, q, strict=True):
        # display() writes transforms to robot_view.viewer; point it at the keyframe instead.
        with animation.at_frame(viewer, t_i) as frame:
            robot_view.viewer = frame
            robot_view.display(q_i)
            pin.framesForwardKinematics(robot.model, tcp_data, q_i)
            frame["tcp"].set_transform(tcp_data.oMf[tcp_id].homogeneous)
    robot_view.viewer = viewer
    viewer.set_animation(animation, repetitions=math.inf)
