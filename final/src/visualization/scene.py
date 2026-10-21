"""MeshCat drawing in the arm's world frame: the viewer, the arm, tool frames and clouds.

Every view draws to one long-lived server, so an open browser tab receives the next view
without a refresh.
"""

import subprocess
import sys
import time

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import zmq
from matplotlib import colormaps
from pinocchio.visualize import MeshcatVisualizer

from core.agx_pinocchio import AgxPinocchio

ZMQ_URL = "tcp://127.0.0.1:6000"
FRAME_LENGTH = 0.012  # m, a drawn tool frame's axes
POINT_SIZE = 0.004  # m


def _server_answers() -> bool:
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 500)
    socket.connect(ZMQ_URL)
    try:
        socket.send(b"url")
        socket.recv()
        return True
    except zmq.Again:
        return False
    finally:
        socket.close()


def open_viewer() -> meshcat.Visualizer:
    """A client of the long-lived server, starting it detached if none answers; the scene cleared."""
    if not _server_answers():
        subprocess.Popen(
            [sys.executable, "-m", "meshcat.servers.zmqserver", "--zmq-url", ZMQ_URL],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + 10.0
        while not _server_answers():
            if time.monotonic() > deadline:
                raise RuntimeError(f"no meshcat server answers on {ZMQ_URL}")
    viewer = meshcat.Visualizer(zmq_url=ZMQ_URL)
    viewer.delete()
    print(f"meshcat: {viewer.url()}")
    return viewer


def show_arm(viewer: meshcat.Visualizer, pin: AgxPinocchio, q: np.ndarray) -> MeshcatVisualizer:
    """The URDF's visual meshes at q (6,) rad; display(q) moves them."""
    robot = pin.robot
    arm = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    arm.initViewer(viewer=viewer)
    arm.loadViewerModel()
    arm.display(q)
    return arm


def colors_of(value: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    """(n,) in [0, 1] -> (n, 3) RGB."""
    return colormaps[cmap](np.clip(value, 0.0, 1.0))[:, :3]


def draw_cloud(node, p: np.ndarray, colors: np.ndarray, size: float = POINT_SIZE) -> None:
    """Points p (n, 3) m in colours (n, 3)."""
    node.set_object(g.PointCloud(p.T.astype(np.float32), colors.T.astype(np.float32), size=size))


def draw_frames(node, p: np.ndarray, rot: np.ndarray, colors: np.ndarray) -> None:
    """Tool frames at p (n, 3) m, rot (n, 3, 3): the key's axis (z) long, its x short.

    The x stub is what shows a turn of the key, which moves its tip nowhere.
    """
    z_end = p + FRAME_LENGTH * rot[:, :, 2]
    x_end = p + 0.5 * FRAME_LENGTH * rot[:, :, 0]
    for name, end in (("z", z_end), ("x", x_end)):
        segments = np.stack([p, end], axis=1).reshape(-1, 3)
        node[name].set_object(
            g.LineSegments(
                g.PointsGeometry(
                    segments.T.astype(np.float32), np.repeat(colors, 2, axis=0).T.astype(np.float32)
                ),
                g.LineBasicMaterial(vertexColors=True),
            )
        )


def draw_lines(node, a: np.ndarray, b: np.ndarray, color: int = 0x888888) -> None:
    """A segment from each a (n, 3) to its b (n, 3), m."""
    segments = np.stack([a, b], axis=1).reshape(-1, 3)
    node.set_object(
        g.LineSegments(
            g.PointsGeometry(segments.T.astype(np.float32)), g.LineBasicMaterial(color=color)
        )
    )


def draw_marker(node, color: int, radius: float = 0.008) -> None:
    node.set_object(
        g.Sphere(radius), g.MeshLambertMaterial(color=color, opacity=0.7, transparent=True)
    )


def place(node, p: np.ndarray) -> None:
    node.set_transform(tf.translation_matrix(p))


def draw_phase_bar(node, length: float = 0.2) -> None:
    """A bar whose filled part, set by set_phase_bar, reads the phase from 0 to 1."""
    node["frame"].set_object(
        g.Box([length, 0.006, 0.006]),
        g.MeshLambertMaterial(color=0xCCCCCC, opacity=0.4, transparent=True),
    )
    node["frame"].set_transform(tf.translation_matrix([length / 2, 0, 0]))
    node["fill"].set_object(g.Box([1.0, 0.01, 0.01]), g.MeshLambertMaterial(color=0xFF8000))


def set_phase_bar(node, phi: float, length: float = 0.2) -> None:
    fill = max(phi, 1e-3) * length
    node["fill"].set_transform(
        tf.translation_matrix([fill / 2, 0, 0]) @ np.diag([fill, 1.0, 1.0, 1.0])
    )
