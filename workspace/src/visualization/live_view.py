#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Watch the arm move, in a browser, while something drives it.

This is a separate process on purpose. It reads the UDP stream that
impedance_control/telemetry.py sends and draws it, whether the sender is a real
run of main.py or `visualization simulate` with no hardware at all; the control loop never waits
for a redraw, and killing or restarting this changes nothing about the run.

It imports no hardware module -- only a socket and the kinematics -- so it
cannot command the arm even by accident.

    python -m visualization watch             # then open the URL it prints
    python -m visualization watch --plot both  # ... with the matplotlib windows
    python -m visualization replay             # a synthetic sweep, no arm needed

The 3D arm needs no display: meshcat renders in the host browser. The
commanded-vs-measured traces are matplotlib and do need one, so --plot is
ignored with a warning unless DISPLAY is set. A forwarded X session already works, since --net=host
shares the host loopback where those displays listen; run.sh additionally
forwards the local unix socket, for a session on the machine's own screen.
"""

import collections
import json
import os
import select
import socket
import time

import matplotlib.pyplot as plt
import numpy as np
from impedance_control.telemetry import TELEMETRY_ADDRESS, open_telemetry, send_sample
from kinematics.model import Q_READY

from visualization.meshcat_view import MeshcatArm

JOINTS = (1, 2, 3, 4, 5, 6)

# How often the scene is redrawn. The stream arrives at 100 Hz; drawing every
# sample would be wasted work, since nothing can be seen faster than this.
REDRAW_SECONDS = 1.0 / 30.0

# How often the worst tracking error is printed, so a headless run still says
# something useful.
REPORT_SECONDS = 1.0

# How much travel the trace plot keeps. The window is counted in samples
# received rather than in the sender's own clock: move_to() restarts its elapsed
# time at zero on every leg of a move, so a time-based window would never expire
# and the plot would grow without bound.
#
# One sample is sent per control step, so STREAM_RATE has to match main.py's
# RATE for the window to really be WINDOW_SECONDS long.
WINDOW_SECONDS = 1.0
STREAM_RATE = 100.0
WINDOW_STEPS = int(WINDOW_SECONDS * STREAM_RATE)


def open_listener():
    """
    The receiving end of the telemetry stream.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(TELEMETRY_ADDRESS)
    sock.setblocking(False)

    return sock


def drain(sock):
    """
    Every sample waiting on the socket, oldest first.

    Draining rather than reading one at a time is what keeps the picture
    current: if drawing fell behind, the backlog is skipped through instead of
    replayed in slow motion.
    """
    samples = []

    while select.select([sock], [], [], 0.0)[0]:
        try:
            samples.append(json.loads(sock.recv(65535)))
        except (OSError, ValueError):
            break

    return samples


def _angles(sample, field):
    """
    One {joint: q} dict out of a sample, as a (6,) array. JSON keys are strings.
    """
    values = sample.get(field)

    if values is None:
        return None

    return np.array([values[str(joint)] for joint in JOINTS])


class Traces:
    """
    A rolling commanded-vs-measured plot, one row per joint.

    This is report_tracking()'s printed rows as a picture: the gap between the
    two lines is the sag being tuned out. Six rows stacked share one x axis, so
    a wrist joint's error is read against what the shoulder was doing at the
    same instant.

    The x axis counts samples received, not the sender's clock. move_to()
    measures elapsed time from the start of each leg, so its t jumps back to
    zero between the demo's moves; plotting against it makes the trace fold
    over itself and the window never roll.
    """

    def __init__(self, kin, numbers_only=False, geometry=None):
        self.plt = plt
        self.numbers_only = numbers_only
        self.history = collections.deque(maxlen=WINDOW_STEPS)
        self.step = 0

        if numbers_only:
            self._build_readout()
            self._place(geometry)
            plt.ion()
            plt.show(block=False)
            return

        self.figure, self.axes = plt.subplots(len(JOINTS), 1, figsize=(9.0, 10.0), sharex=True)

        self.commanded_lines = []
        self.measured_lines = []
        self.error_texts = []

        for joint, axis in zip(JOINTS, self.axes, strict=True):
            self.commanded_lines.append(axis.plot([], [], color="#1f77b4", label="commanded")[0])
            self.measured_lines.append(axis.plot([], [], color="#d62728", label="measured")[0])

            # Fixed limits, zero in the middle and the joint's own stop at the
            # edge. Autoscaling instead lets a joint that is holding still fill
            # the panel with numerical noise: j4 sitting at the 4e-06 rad IK
            # residual drew a full-height curve peaking at "3.8", with the 1e-6
            # exponent in a corner label, which reads as a command to pi. A
            # fixed axis says "not moving" at a glance and makes the six panels
            # comparable with each other.
            span = max(abs(kin.q_min[joint - 1]), abs(kin.q_max[joint - 1]))
            axis.set_ylim(-span, span)

            axis.axhline(0.0, color="#888888", linewidth=0.8)

            # The real stops. They are one-sided on j2 (0 .. 3.14) and j3
            # (-2.967 .. 0), so a symmetric axis on its own would imply travel
            # those joints do not have.
            for stop in (kin.q_min[joint - 1], kin.q_max[joint - 1]):
                axis.axhline(stop, color="#c62828", linewidth=0.8, linestyle="--", alpha=0.5)

            # The live error, on the panel it belongs to. The terminal already
            # prints it once a second, but at a glance a gap between two lines
            # is hard to size against a fixed axis, and it is the number being
            # tuned.
            self.error_texts.append(
                axis.text(
                    0.01,
                    0.94,
                    "",
                    transform=axis.transAxes,
                    family="monospace",
                    fontsize=8,
                    va="top",
                    color="#c62828",
                )
            )

            axis.set_ylabel(f"j{joint} [rad]")
            axis.grid(alpha=0.3)

        self.axes[0].legend(loc="upper right", fontsize=8, ncol=2)
        self.axes[-1].set_xlabel(f"Time [s] @ {STREAM_RATE:.0f}Hz")

        self.figure.suptitle("Live Commanded vs Measured Joint Angles for all 6 joints")

        # Leave a strip at the top for the title; tight_layout does not reserve
        # it for suptitle on its own.
        self.figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

        self._place(geometry)

        plt.ion()
        plt.show(block=False)

    def _build_readout(self):
        """
        The numbers-only window: report_tracking()'s rows, kept up to date.

        Useful when the shape of the motion is already understood and the
        question is just how big the error is right now.
        """
        self.figure, axis = plt.subplots(figsize=(4.6, 3.2))

        axis.axis("off")
        self.axes = [axis]

        self.readout = axis.text(
            0.0,
            1.0,
            "",
            transform=axis.transAxes,
            family="monospace",
            fontsize=11,
            va="top",
            ha="left",
        )

        self.figure.suptitle("Live Joint Angles [rad]")
        self.figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    def _place(self, geometry):
        """
        Put the window where asked, so it can sit beside the meshcat tab.

        meshcat draws in a browser and this is a native window, so the two
        cannot share a frame; meshcat 0.3.2 has no text geometry either, so the
        numbers cannot go into the 3D scene. Placing this window next to the
        browser is as close as the two get.

        Only a Tk-backed window can be positioned, and there is none under Agg,
        so failing here is not worth interrupting a run for.
        """
        window = getattr(self.figure.canvas.manager, "window", None)

        if window is None or not hasattr(window, "wm_geometry"):
            return

        try:
            if geometry is None:
                # Right half of the screen, leaving the left for the browser.
                geometry = f"+{window.winfo_screenwidth() // 2}+0"

            window.wm_geometry(geometry)

        except Exception as error:
            print("INFO: could not place the window:", error)

    def add(self, sample):
        q_cmd, q_meas = _angles(sample, "q_cmd"), _angles(sample, "q_meas")

        if q_cmd is None or q_meas is None:
            return

        self.step += 1

        # A bounded deque is the whole rolling window: the oldest sample falls
        # off as the newest arrives, with nothing to expire or reset.
        self.history.append((self.step, q_cmd, q_meas))

    def draw(self):
        if not self.history:
            return

        # The stored counter is monotonic across legs; seconds are just that
        # counter divided by the rate it arrives at. Plotting the raw count
        # under a "Time [s]" label would be the same kind of lie as an
        # autoscaled axis, so the conversion happens here.
        times = np.array([row[0] for row in self.history]) / STREAM_RATE
        commanded = np.array([row[1] for row in self.history])
        measured = np.array([row[2] for row in self.history])

        error = measured[-1] - commanded[-1]

        if self.numbers_only:
            self._draw_readout(times[-1], commanded[-1], measured[-1], error)
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            return

        for i, axis in enumerate(self.axes):
            self.commanded_lines[i].set_data(times, commanded[:, i])
            self.measured_lines[i].set_data(times, measured[:, i])

            # Only the x window moves; y is fixed to the joint's travel. Setting
            # data on the existing lines rather than redrawing the panel is what
            # makes this keep up with a 100 Hz stream.
            axis.set_xlim(times[0], max(times[-1], times[0] + 1.0 / STREAM_RATE))
            self.error_texts[i].set_text(f"err {error[i]:+.4f}")

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def _draw_readout(self, time, commanded, measured, error):
        """
        Fill the numbers-only window: one row per joint, worst error last.
        """
        worst = int(np.argmax(np.abs(error)))

        rows = [f"t + {time:6.2f} s", "", "        cmd       act       err"]

        rows += [
            f" j{JOINTS[i]}  {commanded[i]:+8.4f}  {measured[i]:+8.4f}  {error[i]:+8.4f}"
            for i in range(len(JOINTS))
        ]

        rows += ["", f" worst {error[worst]:+.4f} on j{JOINTS[worst]}"]

        self.readout.set_text("\n".join(rows))


def replay(seconds=20.0, rate=100.0):
    """
    Feed the viewer a synthetic sweep, so it can be checked with no arm.

    The measured angles lag the commanded ones by a little, which is what a
    real compliant joint does and makes the two traces distinguishable.
    """
    sender = open_telemetry()
    start = time.perf_counter()

    print(f"INFO: replaying a {seconds:.0f}s sweep at {rate:.0f} Hz")

    while True:
        t = time.perf_counter() - start

        if t > seconds:
            break

        q_cmd = {joint: Q_READY[joint - 1] + 0.3 * np.sin(t + joint) for joint in JOINTS}
        q_meas = {joint: Q_READY[joint - 1] + 0.3 * np.sin(t + joint - 0.05) for joint in JOINTS}

        send_sample(sender, t, q_cmd, q_meas)
        time.sleep(1.0 / rate)


def _open_panels(kin, args):
    """
    The matplotlib windows asked for, or none when there is no display to put
    them on.

    --plot selects them: "traces" is the curves, "values" the numbers-only
    readout, "both" puts each in its own window. Only the first is placed, at
    the right of the screen; a second lands wherever the window manager puts it.

    False builds the curves, True the readout -- the one flag Traces takes.
    """
    wanted = {"none": [], "traces": [False], "values": [True], "both": [False, True]}[args.plot]

    if wanted and not os.environ.get("DISPLAY"):
        print("INFO: no DISPLAY, so no matplotlib window. The 3D view still works.")
        return []

    return [Traces(kin, numbers_only=numbers, geometry=None) for numbers in wanted]


def _redraw(view, panels, latest):
    """
    Push the newest sample to the 3D view and every open panel.
    """
    view.set_configuration(_angles(latest, "q_meas"))

    for panel in panels:
        panel.draw()


def _report_worst(latest):
    """
    Print the joint furthest from its commanded angle in the newest sample.
    """
    error = _angles(latest, "q_meas") - _angles(latest, "q_cmd")
    worst = int(np.argmax(np.abs(error)))

    print(f"INFO: t+{latest['t']:6.2f}s  worst error {error[worst]:+.4f} rad on j{JOINTS[worst]}")


def _watch(view, panels, sock):
    """
    Drain telemetry until Ctrl+C, redrawing and reporting on their own clocks.
    """
    drawn = reported = 0.0
    latest = None

    try:
        while True:
            for sample in drain(sock):
                latest = sample

                for panel in panels:
                    panel.add(sample)

            now = time.perf_counter()

            if latest is not None and now - drawn >= REDRAW_SECONDS:
                drawn = now
                _redraw(view, panels, latest)

            if latest is not None and now - reported >= REPORT_SECONDS:
                reported = now
                _report_worst(latest)

            time.sleep(REDRAW_SECONDS / 2.0)

    except KeyboardInterrupt:
        print("")
        print("INFO: stopped watching")


def watch_stream(kin, args):
    """
    Watch whatever is streaming telemetry: a hardware run, or the simulator.

    This is the "watch it move" half of the application. It draws only what
    arrives on the socket, so it does not care whether the sender is main.py
    talking to a real arm or `simulate` running the same trajectory offline.
    """
    view = MeshcatArm(kin)
    view.set_configuration(np.zeros(6))

    panels = _open_panels(kin, args)
    sock = open_listener()

    print("INFO: open", view.url())
    print(f"INFO: listening on {TELEMETRY_ADDRESS[0]}:{TELEMETRY_ADDRESS[1]}   Ctrl+C to stop")

    _watch(view, panels, sock)
