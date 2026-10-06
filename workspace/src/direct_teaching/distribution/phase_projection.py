"""Phase labels for taught datapoints, by projection onto a master trajectory.

Hardware-free. The master is one continuous demonstration of the whole task,
recorded with record_joint_angles.py, and its phase is its normalised time
t / T: a pause while pressing or turning keeps its own share of phi, where a
phase by arc length would collapse it to almost nothing.

A datapoint's phase is that of the nearest master sample by peg_tcp pose, in
the product metric of R^3 x SO(3): d^2 = |dp|^2 + (l * theta)^2, theta the
geodesic rotation angle. SE(3) has no bi-invariant metric, so mixing metres and
radians needs the one length l; it is how far a point l from the rotation axis
travels, and l = 0 is position alone, in plain Euclidean metres. Both
normalisations tried before stalled the phase on the arm: dividing each axis by
the datapoints' spread let the wrist's tilt, then the few millimetres of height,
outweigh centimetres of progress along the path. A fixed physical l cannot be
dominated by a near-constant axis.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from core.agx_pinocchio import AgxPinocchio
from direct_teaching.recorder.joint_angle_recorder import load_recording

TCP_FRAME_NAME = "peg_tcp"
# How far ahead of phi the target kernel is centred, in sigma_f: the forward pull.
# At 1, datapoints at the current phase keep e^-1/2 of the peak weight.
LEAD_SCALE = 1.0


@dataclass
class PhaseContext:
    """The datapoints' and the master's TCP poses, and the parameters derived from them."""

    p: np.ndarray  # (M, 3) m, datapoint TCP positions
    quat: np.ndarray  # (M, 4) datapoint TCP orientations, scalar first
    q_master: np.ndarray  # (N, 6) rad
    p_master: np.ndarray  # (N, 3) m
    quat_master: np.ndarray  # (N, 4)
    phi_master: np.ndarray  # (N,)
    rotation_length: float  # m per rad, l of the pose distance
    k: int
    h: float  # m
    dl_dphi: float  # m per unit phase
    sigma_f: float


def tcp_poses(
    pin_model: AgxPinocchio, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:  # (N, 6) rad -> (N, 3) m, (N, 4) scalar-first quaternions
    poses = [pin_model.forward_kinematics(q_i, TCP_FRAME_NAME) for q_i in q]
    p = np.array([p_i for p_i, _ in poses])
    R = np.array([R_i for _, R_i in poses])
    return p, Rotation.from_matrix(R).as_quat(scalar_first=True)


def rotation_angle(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:  # (..., 4) -> (...) rad
    """The geodesic distance on SO(3); |<a, b>| because q and -q are the same rotation."""
    return 2.0 * np.arccos(np.clip(np.abs((quat_a * quat_b).sum(axis=-1)), 0.0, 1.0))


def pose_distance(
    p_a: np.ndarray,  # (..., 3) m
    quat_a: np.ndarray,  # (..., 4)
    p_b: np.ndarray,
    quat_b: np.ndarray,
    rotation_length: float,  # m per rad
) -> np.ndarray:  # (...) m, broadcast
    rotation = rotation_length * rotation_angle(quat_a, quat_b)
    return np.sqrt(((p_a - p_b) ** 2).sum(axis=-1) + rotation**2)


def master_phases(t: np.ndarray) -> np.ndarray:  # (N,) s -> (N,) in [0, 1]
    return t / t[-1]


def project(context: PhaseContext) -> np.ndarray:  # (M,)
    """Phase of the nearest master sample to each datapoint, by pose distance; the first on ties."""
    d = pose_distance(
        context.p[:, None],
        context.quat[:, None],
        context.p_master[None],
        context.quat_master[None],
        context.rotation_length,
    )
    return context.phi_master[np.argmin(d, axis=1)]


def neighbour_distance(d: np.ndarray) -> tuple[int, float]:  # (M, M) pairwise distances
    """k = round(sqrt(M)) and h, the median distance to the k-th nearest neighbour."""
    k = round(np.sqrt(len(d)))
    d = d.copy()
    np.fill_diagonal(d, np.inf)
    return k, float(np.median(np.sort(d, axis=1)[:, k - 1]))


def phase_weights(
    phi_labels: np.ndarray, phi: float, sigma_f: float, sigma_b: float
) -> np.ndarray:  # (M,), sums to 1
    """The target density's weight per datapoint at phase phi.

    A kernel around min(1, phi + lead), lead = LEAD_SCALE * sigma_f, of width
    sigma_f ahead and sigma_b behind.
    """
    d = phi_labels - min(1.0, phi + LEAD_SCALE * sigma_f)
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
    master_path: Path, q: np.ndarray, pin_model: AgxPinocchio, rotation_length: float = 0.0
) -> PhaseContext:  # q (M, 6) rad, the datapoints; rotation_length m per rad
    t_master, q_master = load_recording(master_path)
    (p, quat), (p_master, quat_master) = tcp_poses(pin_model, q), tcp_poses(pin_model, q_master)
    length = rotation_length
    k, h = neighbour_distance(pose_distance(p[:, None], quat[:, None], p[None], quat[None], length))
    # phi spans 1, so the master's length is its length per unit phase.
    steps = pose_distance(p_master[1:], quat_master[1:], p_master[:-1], quat_master[:-1], length)
    dl_dphi = float(steps.sum())
    return PhaseContext(
        p=p,
        quat=quat,
        q_master=q_master,
        p_master=p_master,
        quat_master=quat_master,
        phi_master=master_phases(t_master),
        rotation_length=rotation_length,
        k=k,
        h=h,
        dl_dphi=dl_dphi,
        sigma_f=h / dl_dphi,
    )


def save_phase_labels(path: Path, q: np.ndarray, phi: np.ndarray, rotation_length: float) -> None:
    """q (M, 6) rad, phi (M,) and the l they were labelled with, so a run projects as they did.

    No t, so load_recording still reads it as a datapoint set.
    """
    np.savez(
        path,
        q=np.asarray(q, dtype=float).reshape(-1, 6),
        phi=np.asarray(phi, dtype=float),
        rotation_length=float(rotation_length),
    )
