"""Offline checks of the phase projection: python test_phase_projection.py."""

import tempfile
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from core.agx_pinocchio import AgxPinocchio
from direct_teaching.distribution.phase_projection import (
    LEAD_SCALE,
    PhaseContext,
    coefficient_density,
    master_phases,
    neighbour_distance,
    phase_context,
    phase_weights,
    position_coefficients,
    project,
    rotation_angle,
    save_phase_labels,
)
from direct_teaching.recorder.joint_angle_recorder import load_recording

URDF_PATH = (
    Path(__file__).resolve().parents[2] / "agx_reference/piper/piper/urdf/piper_description.urdf"
)


def _l_shaped_master() -> tuple[np.ndarray, np.ndarray]:  # (N, 3) m, (N,) phases
    """0.05 m along x, then 0.05 m along y, sampled unevenly in time."""
    rng = np.random.default_rng(0)
    n = 60
    S = np.zeros((2 * n, 3))
    S[:n, 0] = np.linspace(0.0, 0.05, n)
    S[n:, 0] = 0.05
    S[n:, 1] = np.linspace(0.05 / n, 0.05, n)
    t = np.concatenate([[0.0], np.cumsum(rng.uniform(0.5, 1.5, 2 * n - 1))])
    return S, master_phases(t)


def _context(p: np.ndarray, p_master: np.ndarray, phi_master: np.ndarray) -> PhaseContext:
    """Positions only, every orientation the identity."""
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (len(p), 1))
    quat_master = np.tile([1.0, 0.0, 0.0, 0.0], (len(p_master), 1))
    return PhaseContext(
        p=p,
        quat=quat,
        q_master=np.zeros((len(p_master), 6)),
        p_master=p_master,
        quat_master=quat_master,
        phi_master=phi_master,
        rotation_length=0.06,
        k=0,
        h=0.0,
        dl_dphi=0.0,
        sigma_f=0.0,
    )


def test_master_samples_project_to_their_own_phase() -> None:
    S_master, phi_master = _l_shaped_master()
    phi = project(_context(S_master, S_master, phi_master))
    np.testing.assert_array_equal(phi, phi_master)


def test_noisy_points_project_near_their_true_phase() -> None:
    S_master, phi_master = _l_shaped_master()
    rng = np.random.default_rng(1)
    # Well below the sample spacing, about 0.85 mm on both legs.
    S = S_master + 1e-4 * rng.standard_normal(S_master.shape)
    index = np.searchsorted(phi_master, project(_context(S, S_master, phi_master)))
    assert np.abs(index - np.arange(len(S_master))).max() <= 2


def test_rotation_angle_is_the_geodesic_distance_on_so3() -> None:
    rotations = Rotation.random(40, random_state=4)
    R, quat = rotations.as_matrix(), rotations.as_quat(scalar_first=True)
    expected = [np.linalg.norm(pin.log3(R[i].T @ R[i + 1])) for i in range(39)]
    np.testing.assert_allclose(rotation_angle(quat[:-1], quat[1:]), expected, atol=1e-7)
    # q and -q are the same rotation.
    np.testing.assert_allclose(rotation_angle(quat[:-1], -quat[1:]), expected, atol=1e-7)


def test_phase_weights_peak_one_lead_ahead() -> None:
    _, phi_master = _l_shaped_master()
    w = phase_weights(phi_master, 0.3, 0.05, 0.05)
    assert np.isclose(w.sum(), 1.0)
    assert np.argmax(w) == np.argmin(np.abs(phi_master - (0.3 + LEAD_SCALE * 0.05)))


def test_neighbour_distance_on_a_regular_line() -> None:
    # 16 points 1 apart: k = 4. The 4th-nearest distance is 4, 3 at the two ends and
    # one in, and 2 for the twelve interior points, so the median is 2.
    x = np.arange(16.0)
    k, h = neighbour_distance(np.abs(x[:, None] - x[None, :]))
    assert k == 4 and h == 2.0


def test_coefficient_density_matches_the_cosine_series() -> None:
    # c_000 = 1 and c_100 = a stand for p(x, y, z) = 1 + 2 a cos(pi x), analytically.
    c = np.zeros((4, 4, 4))
    c[0, 0, 0], c[1, 0, 0] = 1.0, 0.3
    axis = np.linspace(0.0, 1.0, 7)
    expected = (1 + 0.6 * np.cos(np.pi * axis))[:, None, None] * np.ones((1, 7, 7))
    np.testing.assert_allclose(coefficient_density(c, axis), expected, atol=1e-12)


def test_position_coefficients_have_unit_mass_and_match_a_direct_sum() -> None:
    rng = np.random.default_rng(3)
    X, w = rng.uniform(0.2, 0.8, (30, 3)), rng.uniform(0.0, 1.0, 30)
    w /= w.sum()
    c = position_coefficients(X, w, 5)
    k = (1, 2, 3)
    direct = sum(w_j * np.prod(np.cos(np.pi * x_j * k)) for x_j, w_j in zip(X, w, strict=True))
    assert np.isclose(c[0, 0, 0], 1.0) and np.isclose(c[k], direct)
    # Mass of the density by the midpoint rule: every k > 0 mode integrates to zero.
    axis = (np.arange(40) + 0.5) / 40
    assert np.isclose(coefficient_density(c, axis).mean(), 1.0)


def test_datapoints_on_the_master_get_its_phase() -> None:
    """End to end through files: a master in joint space, datapoints taken from its own samples."""
    pin_model = AgxPinocchio(str(URDF_PATH))
    q_start, q_end = (
        np.array([0.0, 1.0, -1.0, 0.0, 0.5, 0.0]),
        np.array([0.4, 1.4, -1.3, 0.3, 0.2, 0.6]),
    )
    # Still ends, which load_recording trims, around 200 samples of motion.
    q = np.vstack(
        [np.tile(q_start, (20, 1)), np.linspace(q_start, q_end, 200), np.tile(q_end, (20, 1))]
    )
    t = 0.01 * np.arange(len(q))
    with tempfile.TemporaryDirectory() as directory:
        master_path = Path(directory) / "joint_angles_master.npz"
        np.savez(master_path, t=t, q=q)
        q_points = q[20:220:9]
        context = phase_context(master_path, q_points, pin_model, rotation_length=0.06)
        phi = project(context)
        # The motion starts at q_start itself, so sample 20 is the last still one, which the
        # trim keeps as its first: sample 20 + 9 i becomes master sample 9 i.
        np.testing.assert_allclose(phi, context.phi_master[9 * np.arange(len(q_points))])
        labelled_path = Path(directory) / "datapoints_phase.npz"
        save_phase_labels(labelled_path, q_points, phi, context.rotation_length)
        _, q_loaded = load_recording(labelled_path)
        np.testing.assert_array_equal(q_loaded, q_points)
        with np.load(labelled_path) as data:
            assert float(data["rotation_length"]) == 0.06
    assert context.sigma_f > 0


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: ok")
