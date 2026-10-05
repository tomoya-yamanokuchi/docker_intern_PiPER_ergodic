"""Phase labels for taught datapoints, by projection onto a master trajectory.

Hardware-free. The master is one continuous demonstration of the whole task,
recorded with record_joint_angles.py, and its phase is its normalised time
t / T: a pause while pressing or turning keeps its own share of phi, where a
phase by arc length would collapse it to almost nothing.

A datapoint's phase is that of the nearest master sample in the E2T2 state
[p, Log_mu(quat)] of peg_tcp, each axis divided by its standard deviation over
the datapoints so that neither metres nor radians win by their unit.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from core.agx_pinocchio import AgxPinocchio
from direct_teaching.distribution.pose_distribution import PoseDistribution
from direct_teaching.recorder.joint_angle_recorder import load_recording

TCP_FRAME_NAME = "peg_tcp"


@dataclass
class PhaseContext:
    """The datapoints and the master in one state space, and the parameters derived from them."""

    p: np.ndarray  # (M, 3) m, datapoint TCP positions
    S: np.ndarray  # (M, 6) datapoint states
    q_master: np.ndarray  # (N, 6) rad
    p_master: np.ndarray  # (N, 3) m
    S_master: np.ndarray  # (N, 6)
    phi_master: np.ndarray  # (N,)
    scale: np.ndarray  # (6,) per-axis std of the datapoint states
    k: int
    h: float  # scaled state units
    dl_dphi: float  # scaled state units per unit phase
    sigma_f: float


def tcp_poses(
    pin_model: AgxPinocchio, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:  # (N, 6) rad -> (N, 3) m, (N, 4) scalar first
    poses = [pin_model.forward_kinematics(q_i, TCP_FRAME_NAME) for q_i in q]
    p = np.array([p_i for p_i, _ in poses])
    rotations = np.array([R_i for _, R_i in poses])
    return p, Rotation.from_matrix(rotations).as_quat(scalar_first=True)


def pose_states(p: np.ndarray, quaternions: np.ndarray, mu: np.ndarray) -> np.ndarray:  # (N, 6)
    """[p, Log_mu(quat)], unscaled: m and half-angle rad."""
    return np.hstack([p, [PoseDistribution.quaternion_log(q, mu) for q in quaternions]])


def master_phases(t: np.ndarray) -> np.ndarray:  # (N,) s -> (N,) in [0, 1]
    return t / t[-1]


def project(
    S: np.ndarray,  # (M, 6)
    S_master: np.ndarray,  # (N, 6)
    phi_master: np.ndarray,  # (N,)
    scale: np.ndarray,  # (6,)
) -> np.ndarray:  # (M,)
    """Phase of the nearest master sample to each state, in the scaled metric; the first on ties."""
    d2 = (((S[:, None, :] - S_master[None, :, :]) / scale) ** 2).sum(axis=2)
    return phi_master[np.argmin(d2, axis=1)]


def neighbour_distance(X: np.ndarray) -> tuple[int, float]:  # (M, d) scaled states
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
    p, quaternions = tcp_poses(pin_model, q)
    p_master, quaternions_master = tcp_poses(pin_model, q_master)
    # The datapoints' mean orientation, for the master as well: one tangent space for both.
    mu = PoseDistribution.quaternion_mean(quaternions)
    S = pose_states(p, quaternions, mu)
    S_master = pose_states(p_master, quaternions_master, mu)
    scale = S.std(axis=0)
    k, h = neighbour_distance(S / scale)
    # phi spans 1, so the master's scaled length is its length per unit phase.
    dl_dphi = float(np.linalg.norm(np.diff(S_master / scale, axis=0), axis=1).sum())
    return PhaseContext(
        p=p,
        S=S,
        q_master=q_master,
        p_master=p_master,
        S_master=S_master,
        phi_master=master_phases(t_master),
        scale=scale,
        k=k,
        h=h,
        dl_dphi=dl_dphi,
        sigma_f=h / dl_dphi,
    )


def save_phase_labels(path: Path, q: np.ndarray, phi: np.ndarray) -> None:
    """q (M, 6) rad and phi (M,). No t, so load_recording still reads it as a datapoint set."""
    np.savez(path, q=np.asarray(q, dtype=float).reshape(-1, 6), phi=np.asarray(phi, dtype=float))
