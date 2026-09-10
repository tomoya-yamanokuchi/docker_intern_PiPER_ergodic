#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Draw PiPER configurations as a 3D skeleton, to check the model by eye.

The round-trip test compares forward kinematics against forward kinematics, so a
frame convention that is wrong but self-consistent passes it. Looking at the arm
is what catches that.

Everything here reaches the host: a PNG lands on the host because the workspace
is bind-mounted, and both interactive modes are served over HTTP, which the host
browser can open because the container runs with --net=host. That same
--net=host also lets a native window work when DISPLAY names a TCP display on
loopback, as a forwarded X session does -- the container is not headless.

    python -m visualization pose ready              # PNG
    python -m visualization pose branches --web     # matplotlib, :8988
    python -m visualization pose ready --meshcat    # real meshes, :7000

--meshcat draws the actual STL meshes rather than a line skeleton, and is the
one to reach for when the question is whether the model looks like the arm.

Or give a TCP pose in metres and radians and see what the solver makes of it,
which is the quickest way to find out whether something is reachable and what
posture it needs:

    python -m visualization ik 0.30,0.05,0.15,0,3.1416,0
    python -m visualization ik 0.25,0,0.08,0,3.1416,0 --tcp 0.1358 --web
"""

import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from kinematics.dh import fk_dh
from kinematics.fk import fk, joint_frames, pose_error
from kinematics.ik import ik
from kinematics.ik_closed import ik_closed, wrist_centre
from kinematics.model import Q_READY
from kinematics.transform import Transform, rpy_to_matrix

from visualization.meshcat_view import MeshcatArm

AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")  # x, y, z

# How long each configuration is held when --meshcat has several to show,
# which only --pose branches produces. Long enough to read the posture.
BRANCH_SECONDS = 2.0

# Rendered plots go here. .gitignore already covers output/, so they cannot be
# committed by accident.
OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))


def _triad(ax, M, length):
    """
    Draw the x, y, z axes of a frame as three coloured segments.
    """
    for axis, color in enumerate(AXIS_COLORS):
        end = M.translation + length * M.rotation[:, axis]
        ax.plot(*zip(M.translation, end, strict=True), color=color, linewidth=1.4)


def _equal_aspect(ax, points, pad=0.05):
    """
    Force equal scaling on all three axes.

    Matplotlib's 3D default stretches each axis to the data range, which would
    make a correct model look wrong and vice versa.
    """
    points = np.asarray(points)
    centre = (points.max(axis=0) + points.min(axis=0)) / 2.0
    span = (points.max(axis=0) - points.min(axis=0)).max() / 2.0 + pad

    ax.set_xlim(centre[0] - span, centre[0] + span)
    ax.set_ylim(centre[1] - span, centre[1] + span)
    ax.set_zlim(centre[2] - span, centre[2] + span)
    ax.set_box_aspect((1.0, 1.0, 1.0))

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")


def plot_config(kin, q, ax=None, label=None, color="#333333", annotate=("triads", "wrist")):
    """
    Draw one configuration: the link skeleton, frame triads and the wrist.

    annotate selects the overlays drawn on top of the skeleton, any of "triads"
    and "wrist"; pass () for the bare skeleton, which is what overlaying many
    branches at once wants. With "wrist", the joint 4, 5 and 6 axes are extended
    through the wrist centre. They should all pass through the one highlighted
    point -- that is the spherical-wrist property the closed-form solver
    depends on.
    """
    if ax is None:
        ax = _new_axes()

    frames = joint_frames(kin, q)
    points = np.array([[0.0, 0.0, 0.0]] + [M.translation for M in frames])

    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color=color,
        linewidth=2.5,
        marker="o",
        markersize=3.5,
        label=label,
    )

    if "triads" in annotate:
        for M in frames[:-1]:
            _triad(ax, M, 0.035)
        _triad(ax, frames[-1], 0.07)

    if "wrist" in annotate:
        centre = frames[3].translation
        ax.scatter(*centre, s=70, facecolor="none", edgecolor="#ff7f0e", linewidth=1.8, zorder=5)

        for M in frames[3:6]:
            axis = M.rotation[:, 2]
            ax.plot(
                *zip(centre - 0.075 * axis, centre + 0.075 * axis, strict=True),
                color="#ff7f0e",
                linewidth=0.9,
                linestyle="--",
            )

    _equal_aspect(ax, points)

    return ax


def plot_configs(kin, configs, labels=None, ax=None, **kwargs):
    """
    Overlay several configurations, each in its own colour.
    """
    if ax is None:
        ax = _new_axes()

    palette = matplotlib.colormaps["tab10"]

    for i, q in enumerate(configs):
        plot_config(
            kin,
            q,
            ax=ax,
            color=palette(i % 10),
            label=None if labels is None else labels[i],
            **kwargs,
        )

    if labels is not None:
        ax.legend(loc="upper left", fontsize=8)

    return ax


def plot_ik(kin, target, q=None, ax=None):
    """
    Draw a solved configuration together with the pose that was asked for.

    The requested frame is drawn as a long triad; if the solve is right it is
    hidden underneath the arm's own end-effector triad. q may be None for a
    target the solver could not reach, in which case only the target is drawn --
    seeing where it sits is usually enough to explain the failure.
    """
    if q is None:
        if ax is None:
            ax = _new_axes()
        _equal_aspect(ax, np.vstack([np.zeros(3), target.translation]))
    else:
        ax = plot_config(kin, q, ax=ax, label="solution")

    _triad(ax, target, 0.12)
    ax.scatter(*target.translation, s=40, color="#9467bd", label="target")
    ax.legend(loc="upper left", fontsize=8)

    return ax


def plot_dh_check(kin, q, ax=None):
    """
    Overlay the flange position from both published DH tables on the URDF model.

    The markers should sit on the end of the skeleton. They disagree by about
    0.08 mm, which is far too small to see -- the point of the picture is that
    an error large enough to matter would be obvious.
    """
    ax = plot_config(kin, q, ax=ax, label="URDF", annotate=("triads",))

    for convention, marker in (("standard", "x"), ("modified", "+")):
        p = fk_dh(np.asarray(q, dtype=float), convention)[:3, 3]
        ax.scatter(*p, s=110, marker=marker, linewidth=1.8, label=f"DH {convention}")

    ax.legend(loc="upper left", fontsize=8)

    return ax


def _new_axes():
    figure = plt.figure(figsize=(7.5, 6.5))

    return figure.add_subplot(projection="3d")


def _parse_target(text):
    """
    Parse "x,y,z,roll,pitch,yaw" into a Transform. Metres and radians.
    """
    values = [float(v) for v in text.replace(" ", "").split(",")]

    if len(values) != 6:
        raise ValueError(
            f"--target wants 6 comma-separated numbers (x,y,z,roll,pitch,yaw), got {len(values)}"
        )

    return Transform(rpy_to_matrix(*values[3:]), np.array(values[:3]))


def _output_path(name):
    """
    Where a --save argument ends up.

    A bare filename goes under OUTPUT_DIR; a name carrying a directory is used
    as given, for the occasional one-off somewhere else.
    """
    if os.path.dirname(name):
        return name

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    return os.path.join(OUTPUT_DIR, name)


def _resolve_configs(kin, args):
    """
    What --target / --pose ask for: the configurations to draw and the pose
    they were asked for, or None when no pose was named.

    Both renderers go through this, so "what does --pose branches mean" is
    answered in one place and they cannot drift apart.
    """
    if args.target is not None:
        target = _parse_target(args.target)
        q = ik(kin, target, verbose=True)

        return ([] if q is None else [q]), target

    if args.pose == "branches":
        target = fk(kin, ik(kin, fk(kin, Q_READY)))

        return ik_closed(kin, target, limits_only=False), target

    return [np.zeros(6) if args.pose == "zero" else Q_READY], None


def _show_meshcat(kin, args):
    """
    Serve the requested pose in the browser until Ctrl+C.

    meshcat's server is a child of this process, so the wait at the end is what
    keeps the page alive -- it is not a delay. Several configurations, which is
    what --pose branches gives, are shown in turn on the one arm.
    """
    configs, target = _resolve_configs(kin, args)

    view = MeshcatArm(kin)

    if target is not None:
        view.set_target(target)

    if not configs:
        print("INFO: no solution, showing the target on its own")
    elif len(configs) > 1:
        print(f"INFO: cycling {len(configs)} configurations every {BRANCH_SECONDS:.0f}s")

    print("INFO: open", view.url(), "   Ctrl+C to stop serving")

    try:
        while True:
            for q in configs:
                view.set_configuration(q)
                time.sleep(BRANCH_SECONDS)

            if not configs:
                time.sleep(BRANCH_SECONDS)

    except KeyboardInterrupt:
        print("")
        print("INFO: stopped serving")


def _draw_ik(kin, args, configs, target):
    """
    The --target case: the solved configuration, or the target on its own.
    """
    q = configs[0] if configs else None

    ax = plot_ik(kin, target, q)

    if q is None:
        print("INFO: no solution, drawing the target on its own")
        ax.set_title(
            f"UNREACHABLE with tcp offset {args.tcp} m\ntarget {np.round(target.translation, 4)} m"
        )
    else:
        error, _ = pose_error(fk(kin, q), target)
        print("INFO: q [rad] =", np.round(q, 4))
        ax.set_title(
            f"q [rad] = {np.round(q, 3)}\n"
            f"tcp {np.round(target.translation, 4)} m, "
            f"reached to {error * 1e6:.2f} um"
        )


def _draw_branches(kin, configs, target):
    """
    The --pose branches case: every closed-form solution for the one pose.
    """
    ax = plot_configs(kin, configs, [f"branch {i}" for i in range(len(configs))], annotate=())
    _triad(ax, target, 0.12)
    ax.set_title(f"{len(configs)} closed-form branches for one pose")


def _draw_pose(kin, args, configs):
    """
    The named-pose case, with the DH overlay when --dh is given.
    """
    q = configs[0]

    ax = plot_dh_check(kin, q) if args.dh else plot_config(kin, q)
    pose = fk(kin, q)

    ax.set_title(
        f"q = {np.round(q, 3)}\n"
        f"tcp = {np.round(pose.translation, 4)}   "
        f"wrist centre = {np.round(wrist_centre(kin, pose), 4)}"
    )


def show_static(kin, args):
    """
    Draw one pose, or an IK target, as meshcat / an interactive plot / a PNG.

    This is the "look at the model" half of the application: nothing is running
    and no arm is attached, the configurations come from the model itself.
    """
    if args.meshcat:
        _show_meshcat(kin, args)
        return

    # Selecting the backend after pyplot is imported is supported from
    # matplotlib 3.x, and no figure exists yet for the switch to close.
    matplotlib.use("WebAgg" if args.web else "Agg")

    configs, target = _resolve_configs(kin, args)
    name = f"{args.pose}.png"

    if args.target is not None:
        name = "ik.png"
        _draw_ik(kin, args, configs, target)
    elif args.pose == "branches":
        _draw_branches(kin, configs, target)
    else:
        _draw_pose(kin, args, configs)

    # Saving is the default: without a display, a run that renders nothing at
    # all is never what was wanted.
    if args.save or not args.web:
        path = _output_path(args.save or name)
        plt.savefig(path, dpi=130, bbox_inches="tight")
        print(f"INFO: wrote {path}")

    if args.web:
        plt.show()
