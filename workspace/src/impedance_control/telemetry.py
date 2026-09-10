#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Stream one sample per control step to whatever is watching, over UDP.

The point is that watching must never be able to disturb the arm. A UDP
datagram sent to a socket nobody is listening on is simply dropped: sendto()
does not block, does not raise and does not queue, so the command loop runs at
exactly the same rate whether or not a viewer is attached, and a viewer can be
started, killed and restarted in the middle of a trajectory. The cost of a lost
packet is one frame of plot.

The socket is deliberately left unconnected. connect() on a UDP socket asks the
kernel to report ICMP port-unreachable as ECONNREFUSED on the *next* send, which
would turn "no viewer running" into an exception inside the control loop --
exactly the coupling this is meant to avoid.

the visualization application is the other end. Angles are radians, as everywhere else.
"""

import json
import socket

# One destination, named in one place, so the sender and the viewer cannot
# disagree about where the stream goes. Localhost is enough: run.sh uses
# --net=host, so the container and the host share the loopback interface.
TELEMETRY_ADDRESS = ("127.0.0.1", 9870)


def open_telemetry():
    """
    A socket for send_sample(). Nothing is contacted, so this cannot fail.
    """
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_sample(sock, t, q_cmd, q_meas, qdot_ref=None, tau_ff=None):
    """
    Send one control step: the time [s] and the {joint: value} dicts around it.

    Commanded and measured angles travel together because the gap between them
    is the whole quantity being tuned -- the same reason report_tracking()
    prints them as adjacent rows.

    A None sock means telemetry is switched off, which is the normal case for a
    run nobody is watching.
    """
    if sock is None:
        return

    sample = {"t": t, "q_cmd": q_cmd, "q_meas": q_meas, "qdot_ref": qdot_ref, "tau_ff": tau_ff}

    sock.sendto(json.dumps(sample).encode(), TELEMETRY_ADDRESS)
