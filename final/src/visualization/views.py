"""The views of every component, in MeshCat: master, teaching, labelling, a live run, a finished run.

The two live views, TeachingView and RunView, never draw in the caller's thread. A
MeshCat message is a zmq round trip that costs about a control period when calls are a
cycle apart, so update() only stores the newest state and a daemon thread draws it.
Each live view owns its own AgxPinocchio, because the controller's is scratch space it
writes every cycle.

The three offline views are animations: the scrubber under MeshCat's Animations panel
is the slider, reading the phase for the master and the labels, and seconds for a run.
"""

import math
import threading
import time
from pathlib import Path

import meshcat
import numpy as np
from matplotlib.figure import Figure
from meshcat.animation import Animation
from scipy.spatial.transform import Rotation as R

from core.agx_pinocchio import AgxPinocchio
from method.law import sliding_weights
from method.pose import ROTATION_LENGTH, to_metric
from method.task import Task, cube, kernel_width, phase_label, resample, trim_dwell
from robot import TCP_FRAME, URDF_PATH, tool_poses
from visualization.scene import (
    colors_of,
    draw_cloud,
    draw_frames,
    draw_lines,
    draw_marker,
    draw_phase_bar,
    open_viewer,
    place,
    set_phase_bar,
    show_arm,
)

PHASE_FRAMES = 101  # frames of a phase animation
KDE_POINTS = 3000  # draws from the teaching view's kernel density
TRAIL_EVERY = 4  # a live run's trail keeps every this-many-th update
IDLE = 0.01  # s, a drawing thread's poll while nothing is pending
RUN_FRAME = 0.25  # s between a finished run's frames
MASTER_FRAMES = 21  # key frames drawn along the master while teaching
GUIDE_AHEAD = 0.1  # phase the second guide key leads by, so a turn shows before it is due


def datapoint_rotations(task: Task) -> np.ndarray:  # (N_P, 3, 3)
    return R.from_rotvec(task.datapoints[:, 3:] / ROTATION_LENGTH).as_matrix() @ task.R0


def draw_master(viewer: meshcat.Visualizer, master_p: np.ndarray, phi: np.ndarray) -> None:
    """The master's tool path (n, 3) m, coloured by phase from dark (0) to yellow (1)."""
    draw_cloud(viewer["master"], master_p, colors_of(phi), size=0.003)


def _animate_arm(
    viewer, arm, pin: AgxPinocchio, keys: np.ndarray, qs: np.ndarray, per_frame=None
) -> Animation:
    """An animation with the arm at qs[i] and a marker on the tool at keys[i]; per_frame(frame, i) adds more."""
    animation = Animation(default_framerate=1)
    for i, (key, q) in enumerate(zip(keys, qs, strict=True)):
        with animation.at_frame(viewer, key) as frame:
            arm.viewer = frame  # display() writes to arm.viewer; point it at the keyframe
            arm.display(q)
            frame["tool"].set_transform(_tool_transform(pin, q))
            if per_frame is not None:
                per_frame(frame, i)
    arm.viewer = viewer
    return animation


def _tool_transform(pin: AgxPinocchio, q: np.ndarray) -> np.ndarray:  # (4, 4)
    p, rot = pin.forward_kinematics(q, TCP_FRAME)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot, p
    return T


def _draw_tool(viewer: meshcat.Visualizer) -> None:
    """The tool's own frame, drawn at the origin and moved by _tool_transform."""
    draw_frames(
        viewer["tool/axes"], np.zeros((1, 3)), np.eye(3)[None] * 3, np.array([[1.0, 0.3, 0.0]])
    )


def show_master(master: np.ndarray, q_master: np.ndarray, s: np.ndarray) -> None:
    """The arm along the master (resampled), keyed at the phase."""
    pin, viewer = AgxPinocchio(str(URDF_PATH)), open_viewer()
    phi = s / s[-1]
    draw_master(viewer, master[:, :3], phi)
    arm = show_arm(viewer, pin, q_master[0])
    _draw_tool(viewer)
    pick = np.linspace(0, len(phi) - 1, PHASE_FRAMES).round().astype(int)
    viewer.set_animation(_animate_arm(viewer, arm, pin, phi[pick], q_master[pick]), play=False)
    print(f"master: {s[-1]:.3f} m in the pose metric; scrub the animation to move along the phase")


def show_labels(task: Task) -> None:
    """Datapoints coloured by their label, a line to the master sample each is nearest, and,
    over the phase, the sliding target's weights (yellow most, dark least) with the arm on
    the master at that phase."""
    pin, viewer = AgxPinocchio(str(URDF_PATH)), open_viewer()
    phi_m, P = task.phi_master, task.datapoints[:, :3]
    draw_master(viewer, task.master[:, :3], phi_m)
    draw_frames(viewer["labels/frames"], P, datapoint_rotations(task), colors_of(task.phi_p))
    nearest = np.linalg.norm(task.datapoints[:, None] - task.master[None], axis=2).argmin(axis=1)
    draw_lines(viewer["labels/nearest"], P, task.master[nearest, :3])
    arm = show_arm(viewer, pin, task.q_master[0])
    _draw_tool(viewer)

    phis = np.linspace(0.0, 1.0, PHASE_FRAMES)
    for i, phi in enumerate(phis):
        w = sliding_weights(phi, task.phi_p, task.sigma_f)
        draw_cloud(viewer[f"weights/{i}"], P, colors_of(w / w.max()), size=0.006)
        viewer[f"weights/{i}"].set_property("visible", i == 0)

    def show_only(frame, i):  # visibility steps between keys, so neighbours need keys alone
        for j in range(max(i - 1, 0), min(i + 2, PHASE_FRAMES)):
            frame[f"weights/{j}"].set_property("visible", "boolean", j == i)

    qs = task.q_master[np.searchsorted(phi_m, phis).clip(0, len(phi_m) - 1)]
    viewer.set_animation(_animate_arm(viewer, arm, pin, phis, qs, show_only), play=False)
    print(
        f"{len(P)} datapoints, h = {task.h:.4f} m, one resolvable width {task.sigma_f:.3f} in phase"
    )


class TeachingView:
    """The arm as it is backdriven, every datapoint as a tool frame, and the kernel density
    refitted on each change: draws from it, coloured from the datapoints' own range.

    The master is drawn too, its path and key frames coloured by phase; a fan of x stubs at
    one place is the turn. Two guide keys follow the arm: orange is the master's pose at the
    arm's phase (its nearest master sample, as label.py will label it), green the master's
    pose GUIDE_AHEAD further on.
    """

    def __init__(self, q_master: np.ndarray):
        self.pin = AgxPinocchio(str(URDF_PATH))
        self.viewer = open_viewer()
        p, rot = tool_poses(self.pin, q_master)
        self.R0 = rot[0]
        y = to_metric(p, rot, self.R0)
        keep = trim_dwell(y)
        self.master, self.q_master, s = resample(y[keep], q_master[keep])
        self.phi = s / s[-1]
        draw_master(self.viewer, self.master[:, :3], self.phi)
        pick = np.linspace(0, len(self.phi) - 1, MASTER_FRAMES).round().astype(int)
        draw_frames(self.viewer["master_frames"], *tool_poses(self.pin, self.q_master[pick]),
                    colors_of(self.phi[pick]))  # fmt: skip
        for name, color in (("here", [1.0, 0.5, 0.0]), ("ahead", [0.0, 0.8, 0.3])):
            draw_frames(
                self.viewer[f"guide/{name}"], np.zeros((1, 3)), np.eye(3)[None], np.array([color])
            )
        draw_phase_bar(self.viewer["phase_bar"])
        place(self.viewer["phase_bar"], self.master[0, :3] + [0.0, 0.0, 0.08])
        self.arm = show_arm(self.viewer, self.pin, self.q_master[0])
        _draw_tool(self.viewer)
        self.q, self.points = None, None
        threading.Thread(target=self._draw, daemon=True).start()

    def update(self, q: np.ndarray) -> None:
        self.q = q

    def set_datapoints(self, points: list[np.ndarray]) -> None:
        self.points = list(points)

    def _draw(self) -> None:
        while True:
            q, self.q = self.q, None
            points, self.points = self.points, None
            if points is not None:
                self._draw_datapoints(np.array(points).reshape(-1, 6))
            if q is not None:
                self._draw_arm(q)
            if q is None and points is None:
                time.sleep(IDLE)

    def _phase(self, q: np.ndarray) -> float:
        p, rot = self.pin.forward_kinematics(q, TCP_FRAME)
        return float(phase_label(to_metric(p[None], rot[None], self.R0), self.master, self.phi)[0])

    def _master_at(self, phi: float) -> int:
        return int(np.searchsorted(self.phi, min(phi, 1.0)).clip(0, len(self.phi) - 1))

    def _draw_arm(self, q: np.ndarray) -> None:
        self.arm.display(q)
        self.viewer["tool"].set_transform(_tool_transform(self.pin, q))
        phi = self._phase(q)
        for name, at in (("here", phi), ("ahead", phi + GUIDE_AHEAD)):
            T = _tool_transform(self.pin, self.q_master[self._master_at(at)])
            self.viewer[f"guide/{name}"].set_transform(T)
        set_phase_bar(self.viewer["phase_bar"], phi)

    def _draw_datapoints(self, q: np.ndarray) -> None:
        self.viewer["teaching"].delete()
        if not len(q):
            return
        p, rot = tool_poses(self.pin, q)
        draw_frames(self.viewer["teaching/frames"], p, rot, np.tile([0.0, 0.0, 0.0], (len(p), 1)))
        phi = self._phase(q[-1])
        turn = np.degrees(
            np.linalg.norm(to_metric(p[-1:], rot[-1:], self.R0)[0, 3:]) / ROTATION_LENGTH
        )
        master_turn = np.degrees(
            np.linalg.norm(self.master[self._master_at(phi), 3:]) / ROTATION_LENGTH
        )
        print(f"  newest at phase {phi:.2f}: rotated {turn:.0f} deg from the master's start, "
              f"the master {master_turn:.0f} deg there")  # fmt: skip
        if len(q) < 3:
            return
        y = to_metric(p, rot, rot[0])
        lo, L = cube(y)
        h = kernel_width(y, lo, L)
        rng = np.random.default_rng(0)
        draws = p[rng.integers(len(p), size=KDE_POINTS)] + rng.normal(0.0, h, (KDE_POINTS, 3))
        draw_cloud(
            self.viewer["teaching/density"], draws, np.tile([0.2, 0.4, 1.0], (KDE_POINTS, 1)), 0.002
        )
        print(f"  refitted: h = {h:.4f} m")


class RunView:
    """A run as it happens: the arm, the commanded pose, the phase as a marker on the master
    and as a bar, the datapoints coloured by their weight in the target, and the trail."""

    def __init__(self, task: Task):
        self.task = task
        self.pin = AgxPinocchio(str(URDF_PATH))
        self.viewer = open_viewer()
        draw_master(self.viewer, task.master[:, :3], task.phi_master)
        self.arm = show_arm(self.viewer, self.pin, task.q_master[0])
        _draw_tool(self.viewer)
        draw_marker(self.viewer["phase"], 0xFF8000)
        draw_marker(self.viewer["command"], 0xFF0000, radius=0.004)
        bar = self.viewer["phase_bar"]
        draw_phase_bar(bar)
        place(bar, task.master[0, :3] + [0.0, 0.0, 0.08])
        self.pending, self.trail, self.updates = None, [], 0
        threading.Thread(target=self._draw, daemon=True).start()

    def new_trial(self) -> None:
        self.trail, self.updates = [], 0

    def update(self, q: np.ndarray, y_cmd: np.ndarray, phi: float, w: np.ndarray) -> None:
        self.pending = (q, y_cmd, phi, w)

    def _draw(self) -> None:
        while True:
            pending, self.pending = self.pending, None
            if pending is None:
                time.sleep(IDLE)
                continue
            q, y_cmd, phi, w = pending
            self.arm.display(q)
            T = _tool_transform(self.pin, q)
            self.viewer["tool"].set_transform(T)
            place(self.viewer["command"], y_cmd[:3])
            place(
                self.viewer["phase"],
                [np.interp(phi, self.task.phi_master, m) for m in self.task.master[:, :3].T],
            )
            set_phase_bar(self.viewer["phase_bar"], phi)
            draw_cloud(
                self.viewer["weights"],
                self.task.datapoints[:, :3],
                colors_of(w / w.max()),
                size=0.006,
            )
            self.updates += 1
            if self.updates % TRAIL_EVERY == 0:
                self.trail.append(T[:3, 3])
                draw_cloud(
                    self.viewer["trail"],
                    np.array(self.trail),
                    np.zeros((len(self.trail), 3)),
                    0.002,
                )


def show_run(task: Task, run: dict, trial: int) -> None:
    """One trial of a finished run, keyed in seconds: the arm, the phase marker and bar, and
    the target's weights as they were."""
    pin, viewer = AgxPinocchio(str(URDF_PATH)), open_viewer()
    mine = run["trial"] == trial
    t, q, phi, w = run["t"][mine], run["q"][mine], run["phi"][mine], run["w"][mine]
    pick = np.searchsorted(t, np.arange(0.0, t[-1], RUN_FRAME))
    draw_master(viewer, task.master[:, :3], task.phi_master)
    arm = show_arm(viewer, pin, q[0])
    _draw_tool(viewer)
    draw_marker(viewer["phase"], 0xFF8000)
    draw_phase_bar(viewer["phase_bar"])
    place(viewer["phase_bar"], task.master[0, :3] + [0.0, 0.0, 0.08])
    draw_cloud(viewer["trail"], tool_poses(pin, q[::5])[0], np.zeros((len(q[::5]), 3)), 0.002)
    for i, k in enumerate(pick):
        draw_cloud(
            viewer[f"weights/{i}"], task.datapoints[:, :3], colors_of(w[k] / w[k].max()), size=0.006
        )
        viewer[f"weights/{i}"].set_property("visible", i == 0)

    def per_frame(frame, i):
        k = pick[i]
        frame["phase"].set_transform(
            _translation([np.interp(phi[k], task.phi_master, m) for m in task.master[:, :3].T])
        )
        fill = max(phi[k], 1e-3) * 0.2
        frame["phase_bar/fill"].set_transform(
            _translation([fill / 2, 0, 0]) @ np.diag([fill, 1.0, 1.0, 1.0])
        )
        for j in range(max(i - 1, 0), min(i + 2, len(pick))):
            frame[f"weights/{j}"].set_property("visible", "boolean", j == i)

    viewer.set_animation(
        _animate_arm(viewer, arm, pin, t[pick], q[pick], per_frame), repetitions=math.inf
    )


def _translation(p) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = p
    return T


def plot_run(run: dict, path: Path) -> None:
    """Per trial, the phase, the back edge's retreat and the stall over time, to a PNG."""
    trials = np.unique(run["trial"])
    fig = Figure(figsize=(10, 2.6 * len(trials)))
    for i, trial in enumerate(trials):
        mine = run["trial"] == trial
        ax = fig.add_subplot(len(trials), 1, i + 1)
        ax.plot(run["t"][mine], run["phi"][mine], label="phase")
        ax.plot(run["t"][mine], run["rho"][mine], label="retreat (phase)")
        ax.set(ylabel="phase", ylim=(-0.02, 1.02), title=f"trial {trial + 1}: {run['outcomes'][i]}")
        stall = ax.twinx()
        stall.plot(run["t"][mine], run["stall"][mine], color="0.6", lw=0.8, label="stall")
        stall.set_ylabel("stall (s)")
        ax.legend(loc="upper left")
    ax.set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(path)
