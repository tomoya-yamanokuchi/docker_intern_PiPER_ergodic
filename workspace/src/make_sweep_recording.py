#!/usr/bin/env python3
"""Write a synthetic joint-angle recording that sweeps one joint at a time.

A repeatable stand-in for a hand-taught recording, for judging the friction
feedforward of play_joint_angles.py: each joint reverses direction several times
at a known speed, and Coulomb friction is what makes a reversal lag.

Every segment is a raised cosine, so it starts and ends at rest at the base pose
and the joint never steps in velocity. JointAnglePlayer differentiates the
recording twice over a 0.05 s window, so a triangle wave like the one
identify_friction.py sweeps would turn each corner into an acceleration spike.

Hardware-free. Run from workspace/src:  python make_sweep_recording.py
"""

from datetime import datetime
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

# The pose identify_friction.py measured the friction at, so the coefficients
# fed forward during the replay belong to this part of the workspace.
BASE_POSE = np.array([0.0, 1.1869, -0.8122, 0.0337, 0.5554, 1.431])  # (6,) rad

# Joint limits, pyAgxArm config "joint_limits" for PiperFW.DEFAULT.
JOINT_LIMITS = np.array(
    [
        [-2.6180, 2.6180],
        [0.0, 3.1416],
        [-2.9671, 0.0],
        [-1.7453, 1.7453],
        [-1.2217, 1.2217],
        [-2.0944, 2.0944],
    ]
)  # (6, 2) rad

SWEEP_AMPLITUDE = 0.3  # rad, travel above the base pose
PEAK_SPEED = 0.4  # rad/s; the level every joint tracked cleanly during identification
SWEEP_CYCLES = 3  # per joint
DWELL_S = 0.5  # rest between joints, so each sweep reads separately
# Denser than the 10 Hz of a real recording: the player's central differences use
# a 0.05 s half-step, which a 0.1 s sample interval would alias.
SAMPLE_RATE_HZ = 50.0


def sweep_offset(t: np.ndarray, amplitude: float, freq_hz: float) -> np.ndarray:
    """Raised cosine from 0 up to amplitude and back, freq_hz cycles per second."""
    return 0.5 * amplitude * (1.0 - np.cos(2.0 * np.pi * freq_hz * t))


def build_recording() -> tuple[np.ndarray, np.ndarray]:  # t (N,) s, q (N, 6) rad
    """Base pose throughout, with each joint sweeping up and back in turn."""
    # Peak speed of the raised cosine is pi * f * amplitude.
    freq_hz = PEAK_SPEED / (np.pi * SWEEP_AMPLITUDE)
    segment = np.arange(0.0, SWEEP_CYCLES / freq_hz, 1.0 / SAMPLE_RATE_HZ)
    dwell = np.arange(0.0, DWELL_S, 1.0 / SAMPLE_RATE_HZ)

    blocks = [np.tile(BASE_POSE, (len(dwell), 1))]
    for joint in range(len(BASE_POSE)):
        q = np.tile(BASE_POSE, (len(segment), 1))
        q[:, joint] += sweep_offset(segment, SWEEP_AMPLITUDE, freq_hz)
        blocks.append(q)
        blocks.append(np.tile(BASE_POSE, (len(dwell), 1)))

    q = np.vstack(blocks)
    t = np.arange(len(q)) / SAMPLE_RATE_HZ
    return t, q


def main() -> None:
    t, q = build_recording()

    # This file drives a real arm, so prove the angles are reachable before writing.
    below = q.min(axis=0) < JOINT_LIMITS[:, 0]
    above = q.max(axis=0) > JOINT_LIMITS[:, 1]
    if below.any() or above.any():
        raise ValueError(f"sweep leaves the joint limits: {np.flatnonzero(below | above) + 1}")

    path = OUTPUT_DIR / f"joint_angles_{datetime.now():%Y%m%d_%H%M%S}.npz"
    np.savez(path, t=t, q=q)
    print(f"saved {len(t)} samples over {t[-1]:.1f} s to {path}")
    print(f"sweep {SWEEP_AMPLITUDE} rad, peak {PEAK_SPEED} rad/s, {SWEEP_CYCLES} cycles per joint")


if __name__ == "__main__":
    main()
