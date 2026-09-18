"""Offline checks of PoseDistribution: python test_pose_distribution.py."""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal

from direct_teaching.distribution.pose_distribution import PoseDistribution

RNG_SEED = 0


def _cluster(
    rng: np.random.Generator, centre: np.ndarray, n: int, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = dt * np.arange(n)
    p = centre + 0.005 * rng.standard_normal((n, 3))
    R = (
        Rotation.from_rotvec(0.02 * rng.standard_normal((n, 3))) * Rotation.from_euler("x", 0.4)
    ).as_matrix()
    return t, p, R


def _two_cluster_recording() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)
    # Same sample count, but the operator dwelt three times longer per sample on the slow one.
    fast = _cluster(rng, np.array([0.3, 0.2, 0.2]), 50, dt=0.1)
    slow = _cluster(rng, np.array([0.3, 0.0, 0.2]), 50, dt=0.3)
    # One recording, fast then slow, 0.1 s apart.
    t = np.concatenate([fast[0], fast[0][-1] + 0.1 + slow[0]])
    return t, np.concatenate([fast[1], slow[1]]), np.concatenate([fast[2], slow[2]])


def _two_cluster_distribution() -> PoseDistribution:
    return PoseDistribution(*_two_cluster_recording(), n_components=2)


def test_quaternion_log_is_half_the_rotation_vector() -> None:
    for rotation in Rotation.random(200, random_state=RNG_SEED):
        q = rotation.as_quat(scalar_first=True)
        # scipy's rotvec has angle in [0, pi], the shorter representative.
        expected = rotation.as_rotvec() / 2
        np.testing.assert_allclose(PoseDistribution.quaternion_log(q), expected, atol=1e-9)
        np.testing.assert_allclose(PoseDistribution.quaternion_log(-q), expected, atol=1e-9)


def test_quaternion_log_at_g_is_relative_rotation() -> None:
    rotations = Rotation.random(200, random_state=RNG_SEED)
    bases = Rotation.random(200, random_state=RNG_SEED + 1)
    for rotation, base in zip(rotations, bases, strict=True):
        q, g = rotation.as_quat(scalar_first=True), base.as_quat(scalar_first=True)
        expected = (base.inv() * rotation).as_rotvec() / 2
        np.testing.assert_allclose(PoseDistribution.quaternion_log(q, g), expected, atol=1e-9)


def test_quaternion_exp_inverts_log() -> None:
    rotations = Rotation.random(200, random_state=RNG_SEED)
    bases = Rotation.random(200, random_state=RNG_SEED + 1)
    for rotation, base in zip(rotations, bases, strict=True):
        q, g = rotation.as_quat(scalar_first=True), base.as_quat(scalar_first=True)
        back = PoseDistribution.quaternion_exp(PoseDistribution.quaternion_log(q, g), g)
        # q and -q are the same rotation.
        assert min(np.linalg.norm(back - q), np.linalg.norm(back + q)) < 1e-9
    np.testing.assert_array_equal(PoseDistribution.quaternion_exp(np.zeros(3), g), g)


def test_quaternion_mean_of_symmetric_set_is_its_centre() -> None:
    centre = Rotation.from_euler("xyz", [0.3, -0.7, 1.1]).as_quat(scalar_first=True)
    offsets = [0.2 * s * e for e in np.eye(3) for s in (1.0, -1.0)]
    quaternions = [PoseDistribution.quaternion_exp(v, centre) for v in offsets]
    mean = PoseDistribution.quaternion_mean(np.array(quaternions))
    assert min(np.linalg.norm(mean - centre), np.linalg.norm(mean + centre)) < 1e-9


def test_pose_state_round_trip_and_data_inside_cube() -> None:
    distribution = _two_cluster_distribution()
    _, p, R = _cluster(np.random.default_rng(RNG_SEED), np.array([0.3, 0.1, 0.2]), 20, 0.1)
    for p_i, R_i in zip(p, R, strict=True):
        x = distribution.pose_to_state(p_i, R_i)
        assert np.all((x > 0.0) & (x < 1.0))
        p_back, R_back = distribution.state_to_pose(x)
        np.testing.assert_allclose(p_back, p_i, atol=1e-9)
        np.testing.assert_allclose(R_back, R_i, atol=1e-9)


def test_margin_places_data_extremes() -> None:
    margin = 0.1
    distribution = _two_cluster_distribution()
    _, p, R = _two_cluster_recording()
    X = np.array([distribution.pose_to_state(p_i, R_i) for p_i, R_i in zip(p, R, strict=True)])
    # y spans 0.2 m, above MIN_SPAN, so the data fills [m, 1 - m] / (1 + 2m) exactly.
    edge = margin / (1 + 2 * margin)
    np.testing.assert_allclose([X[:, 1].min(), X[:, 1].max()], [edge, 1 - edge], atol=1e-9)


def test_pdf_is_the_weighted_gaussian_sum() -> None:
    distribution = _two_cluster_distribution()
    np.testing.assert_allclose(distribution.priors.sum(), 1.0)
    X = np.random.default_rng(RNG_SEED).uniform(0.0, 1.0, (100, 6))
    X[:2] = distribution.means  # somewhere the density is not negligible
    expected = sum(
        prior * multivariate_normal(mean, covariance).pdf(X)
        for prior, mean, covariance in zip(
            distribution.priors, distribution.means, distribution.covariances, strict=True
        )
    )
    np.testing.assert_allclose(distribution.pdf(X), expected, rtol=1e-9, atol=0.0)


def test_marginal_pdf_integrates_the_full_pdf() -> None:
    distribution = _two_cluster_distribution()
    kept, integrated = [0, 1, 2, 3], [4, 5]
    axis = np.linspace(-1.0, 2.0, 601)
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    X = distribution.means.copy()  # (2, 6); the kept part is where to compare
    for x in X:
        full = np.empty((len(grid), 6))
        full[:, kept] = x[kept]
        full[:, integrated] = grid
        density = distribution.pdf(full).reshape(len(axis), len(axis))
        expected = np.trapezoid(np.trapezoid(density, axis, axis=1), axis)
        np.testing.assert_allclose(distribution.marginal_pdf(x[kept], kept), expected, rtol=1e-6)


def test_priors_weight_by_time_not_sample_count() -> None:
    distribution = _two_cluster_distribution()
    # The fast cluster covers 50 * 0.1 = 5 s. The slow one covers 49 * 0.3 s plus
    # the median interval, 0.1 s (50 of the 99 intervals are 0.1 s), for its last sample.
    fast_s, slow_s = 5.0, 49 * 0.3 + 0.1
    y_of_component = [distribution.state_to_pose(mean)[0][1] for mean in distribution.means]
    slow = int(np.argmin(y_of_component))
    np.testing.assert_allclose(distribution.priors[slow], slow_s / (slow_s + fast_s), atol=1e-3)
    np.testing.assert_allclose(distribution.priors[1 - slow], fast_s / (slow_s + fast_s), atol=1e-3)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
