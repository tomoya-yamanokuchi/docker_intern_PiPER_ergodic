"""Flange tracking error of a replay, and a figure of it for a report.

Hardware-free: takes the q and q_target the replay loop logged and needs only
the Pinocchio model.
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from core.agx_pinocchio import AgxPinocchio, helper


def compute_tracking_errors(
    pin_model: AgxPinocchio,
    q: np.ndarray,  # (N, 6) rad, measured
    q_target: np.ndarray,  # (N, 6) rad, commanded
    frame_name: str,
) -> tuple[np.ndarray, np.ndarray]:  # (N,) m, (N,) rad
    """Norms of the Cartesian impedance error [p_target - p, log3(R_target R^T)]."""
    position_error = np.empty(len(q))
    orientation_error = np.empty(len(q))
    for i, (q_i, q_target_i) in enumerate(zip(q, q_target, strict=True)):
        p, R = pin_model.forward_kinematics(q_i, frame_name)
        p_target, R_target = pin_model.forward_kinematics(q_target_i, frame_name)
        position_error[i] = np.linalg.norm(p_target - p)
        orientation_error[i] = np.linalg.norm(helper.orientation_error_rotmat(R_target, R))
    return position_error, orientation_error


def save_tracking_error_plot(
    t: np.ndarray,  # (N,) s
    position_error: np.ndarray,  # (N,) m
    orientation_error: np.ndarray,  # (N,) rad
    t_recording: tuple[float, float],  # s, start and end of the recorded part
    path: Path,
) -> None:
    """Two stacked panels sharing t, sized for one IEEE column."""
    fig = Figure(figsize=(3.5, 2.8), layout="constrained")
    ax_p, ax_r = fig.subplots(2, 1, sharex=True)
    for ax, error, label in (
        (ax_p, position_error, "position error [m]"),
        (ax_r, orientation_error, "orientation error [rad]"),
    ):
        ax.plot(t, error, color="#1f5f99", linewidth=1.0)
        for t_mark in t_recording:
            ax.axvline(t_mark, color="0.6", linestyle="--", linewidth=0.8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_ylim(bottom=0.0)
        ax.tick_params(labelsize=7)
        ax.grid(color="0.9", linewidth=0.5)
    ax_r.set_xlabel("t [s]", fontsize=8)
    ax_r.set_xlim(t[0], t[-1])
    fig.suptitle("Flange tracking error during replay", fontsize=9)
    fig.savefig(path, dpi=300)
