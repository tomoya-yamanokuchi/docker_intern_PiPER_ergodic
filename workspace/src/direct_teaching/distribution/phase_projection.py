"""Phase labels for taught datapoints, by projection onto a master trajectory.

Hardware-free. The master is one continuous demonstration of the whole task,
recorded with record_joint_angles.py, and its phase is its normalised time
t / T: a pause while pressing or turning keeps its own share of phi, where a
phase by arc length would collapse it to almost nothing.

A datapoint's phase is that of the nearest master sample by peg_tcp position
alone, in plain Euclidean metres. Both normalisations tried before stalled the
phase on the arm: dividing each axis by the datapoints' spread let the wrist's
tilt, then the few millimetres of height, outweigh centimetres of progress
along the path.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.agx_pinocchio import AgxPinocchio
from direct_teaching.recorder.joint_angle_recorder import load_recording

TCP_FRAME_NAME = "peg_tcp"


@dataclass
class PhaseContext:
    """The datapoints' and the master's TCP positions, and the parameters derived from them."""

    p: np.ndarray  # (M, 3) m, datapoint TCP positions
    q_master: np.ndarray  # (N, 6) rad
    p_master: np.ndarray  # (N, 3) m
    phi_master: np.ndarray  # (N,)
    k: int
    h: float  # m
    dl_dphi: float  # m per unit phase
    sigma_f: float


def tcp_positions(pin_model: AgxPinocchio, q: np.ndarray) -> np.ndarray:  # (N, 6) rad -> (N, 3) m
    return np.array([pin_model.forward_kinematics(q_i, TCP_FRAME_NAME)[0] for q_i in q])


def master_phases(t: np.ndarray) -> np.ndarray:  # (N,) s -> (N,) in [0, 1]
    return t / t[-1]


def project(
    p: np.ndarray,  # (M, 3) m
    p_master: np.ndarray,  # (N, 3) m
    phi_master: np.ndarray,  # (N,)
) -> np.ndarray:  # (M,)
    """Phase of the nearest master sample to each position, in metres; the first on ties."""
    d2 = ((p[:, None, :] - p_master[None, :, :]) ** 2).sum(axis=2)
    return phi_master[np.argmin(d2, axis=1)]


def neighbour_distance(X: np.ndarray) -> tuple[int, float]:  # (M, d)
    """k = round(sqrt(M)) and h, the median distance to the k-th nearest neighbour."""
    k = round(np.sqrt(len(X)))
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return k, float(np.median(np.sort(d, axis=1)[:, k - 1]))


def phase_weights(
    phi_labels: np.ndarray, phi: float, sigma_f: float, sigma_b: float
) -> np.ndarray:  # (M,), sums to 1
    """The target density's weight per datapoint at phase phi.

    A kernel around min(1, phi + lead), lead = sigma_f, of width sigma_f ahead
    and sigma_b behind.
    """
    d = phi_labels - min(1.0, phi + sigma_f)
    sigma = np.where(d > 0, sigma_f, sigma_b)
    w = np.exp(-(d**2) / (2 * sigma**2))
    return w / w.sum()


def position_coefficients(
    X: np.ndarray, w: np.ndarray, K: int
) -> np.ndarray:  # (M, 3) cube positions, (M,) weights summing to 1 -> (K, K, K)
    """W_hat_k = sum_j w_j Phi_k(x_j) over the position axes: the target the ergodic law sees.

    Only the position marginal, since Phi_0 = 1 on every orientation axis.
    """
    phi = np.cos(np.pi * X[:, :, None] * np.arange(K))  # (M, 3, K)
    return np.einsum("m,ma,mb,mc->abc", w, phi[:, 0], phi[:, 1], phi[:, 2], optimize=True)


def coefficient_density(
    c: np.ndarray, axis: np.ndarray
) -> np.ndarray:  # (K, K, K), (G,) -> (G, G, G)
    """The density the coefficients stand for on the grid axis^3 of the unit cube.

    Inverse cosine series, p = sum_k c_k Phi_k / h_k with h_k = 1 for k = 0 and
    1/2 otherwise. Truncated at K modes it rings, so it can go negative.
    """
    k = np.arange(c.shape[0])
    B = np.cos(np.pi * np.outer(axis, k)) / np.where(k == 0, 1.0, 0.5)
    return np.einsum("abc,ia,jb,kc->ijk", c, B, B, B, optimize=True)


def phase_context(
    master_path: Path, q: np.ndarray, pin_model: AgxPinocchio
) -> PhaseContext:  # q (M, 6) rad, the datapoints
    t_master, q_master = load_recording(master_path)
    p, p_master = tcp_positions(pin_model, q), tcp_positions(pin_model, q_master)
    k, h = neighbour_distance(p)
    # phi spans 1, so the master's length is its length per unit phase.
    dl_dphi = float(np.linalg.norm(np.diff(p_master, axis=0), axis=1).sum())
    return PhaseContext(
        p=p,
        q_master=q_master,
        p_master=p_master,
        phi_master=master_phases(t_master),
        k=k,
        h=h,
        dl_dphi=dl_dphi,
        sigma_f=h / dl_dphi,
    )


def save_phase_labels(path: Path, q: np.ndarray, phi: np.ndarray) -> None:
    """q (M, 6) rad and phi (M,). No t, so load_recording still reads it as a datapoint set."""
    np.savez(path, q=np.asarray(q, dtype=float).reshape(-1, 6), phi=np.asarray(phi, dtype=float))
