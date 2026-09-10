#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
The PiPER visualisation application. One tool, several things to look at.

    python -m visualization pose ready              # PNG of the model
    python -m visualization pose branches --web     # matplotlib, :8988
    python -m visualization pose ready --meshcat    # the real STL meshes, :7000
    python -m visualization pose zero --dh          # overlay the published DH tables

    python -m visualization ik 0.30,0.05,0.15,0,3.1416,0     # solve and draw a TCP pose

    python -m visualization watch                   # follow a running main.py
    python -m visualization watch --plot traces     # ... and plot cmd vs measured

    python -m visualization simulate --start zero --target ready         # no arm needed
    python -m visualization simulate --target 0,1.5,-0.9,0,1.0,0         # explicit angles
    python -m visualization replay                  # synthetic sweep, checks the viewer

Everything reaches the host: PNGs land on the host because the workspace is
bind-mounted, and the interactive modes are served over HTTP, which the host
browser can open because the container runs with --net=host. That same
--net=host lets a native matplotlib window work when DISPLAY names a TCP display
on loopback, as a forwarded X session does -- the container is not headless.

`watch` and `simulate` are the same picture from two sources. `watch` listens for
telemetry from a real run; `simulate` generates that telemetry itself from the
model, running the identical trajectory code against SimulatedArm, so a motion
can be seen before it is ever sent to hardware. Start `watch` in one shell and
`simulate` in another to see it move, or run `simulate` alone to check that the
trajectory is generated and inside the joint limits at all.
"""

import argparse

from kinematics.model import PiperKinematics

from visualization.live_view import replay, watch_stream
from visualization.simulate import simulate_trajectory
from visualization.skeleton import show_static

# Every mode takes the TCP offset, because it decides where the tool point is
# drawn and must match whatever the controller is running with.
TCP_HELP = "TCP offset along the flange z axis [m]; 0.1358 puts it at the gripper finger base"

# Both simulate poses accept the same forms, so the help text is written once.
POSE_HELP = "Either zero or ready, or six comma-separated joint angles in radians, j1 first"


def _add_static_options(parser):
    """
    The options shared by the two modes that draw the model.
    """
    parser.add_argument("--tcp", type=float, default=0.0, metavar="M", help=TCP_HELP)
    parser.add_argument(
        "--save",
        default=None,
        metavar="PNG",
        help="filename to write, under workspace/output/ unless it carries a directory",
    )
    parser.add_argument(
        "--meshcat",
        action="store_true",
        help="show the real meshes in the host browser instead of a skeleton; needs no X display",
    )
    parser.add_argument(
        "--web", action="store_true", help="serve an interactive plot on http://localhost:8988"
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    modes = parser.add_subparsers(dest="mode", required=True)

    pose = modes.add_parser("pose", help="draw a named configuration of the model")
    pose.add_argument(
        "pose",
        nargs="?",
        default="ready",
        choices=("zero", "ready", "branches"),
        help="branches draws every closed-form solution for one target",
    )
    pose.add_argument("--dh", action="store_true", help="overlay the published DH tables")
    pose.set_defaults(target=None)
    _add_static_options(pose)

    solve = modes.add_parser("ik", help="solve a TCP pose and draw the result")
    solve.add_argument("target", metavar="X,Y,Z,ROLL,PITCH,YAW", help="metres and radians")
    solve.set_defaults(pose="ik", dh=False)
    _add_static_options(solve)

    follow = modes.add_parser("watch", help="follow telemetry from a run or from simulate")
    follow.add_argument("--tcp", type=float, default=0.0, metavar="M", help=TCP_HELP)
    follow.add_argument(
        "--plot",
        default="none",
        choices=("none", "traces", "values", "both"),
        help="add a matplotlib window: curves, a numbers-only readout, or both. "
        "Needs DISPLAY set to a display this container can reach",
    )

    run = modes.add_parser("simulate", help="travel between two poses with no hardware attached")
    run.add_argument(
        "--start", default="zero", metavar="POSE", help=f"pose to start from. {POSE_HELP}"
    )
    run.add_argument(
        "--target", default="ready", metavar="POSE", help=f"pose to travel to. {POSE_HELP}"
    )
    run.add_argument("--tcp", type=float, default=0.0, metavar="M", help=TCP_HELP)
    run.add_argument(
        "--max-speed",
        type=float,
        default=0.5,
        metavar="RAD_S",
        help="travel speed of the fastest joint",
    )
    run.add_argument("--rate", type=float, default=100.0, metavar="HZ", help="command rate")

    modes.add_parser("replay", help="send a synthetic sweep, to check the viewer with no arm")

    return parser


def main():
    args = _build_parser().parse_args()

    if args.mode == "replay":
        replay()
        return

    kin = PiperKinematics(tcp_offset=args.tcp)

    if args.mode == "watch":
        watch_stream(kin, args)
    elif args.mode == "simulate":
        simulate_trajectory(kin, args)
    else:
        show_static(kin, args)


if __name__ == "__main__":
    main()
