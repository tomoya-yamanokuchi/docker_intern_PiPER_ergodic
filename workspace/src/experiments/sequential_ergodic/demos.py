"""Scripted noisy demonstrations, and the master path mu(s) with its tube width sigma(s)."""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from experiments.sequential_ergodic.scene_a import SceneA


def resample(path: np.ndarray, n: int) -> np.ndarray:  # (M, 2) -> (n, 2)
    """The path at n points equally spaced in arc length, ends included."""
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    s = np.linspace(0.0, arc[-1], n)
    return np.column_stack([np.interp(s, arc, path[:, i]) for i in range(2)])


def scene_a_waypoints(scene: SceneA) -> np.ndarray:  # (4, 2)
    """Start, straight up through gap A, then to the goal."""
    below, above = scene.wall_y - 0.1, scene.wall_y + 0.1
    return np.array([scene.start, (scene.gap_a_x, below), (scene.gap_a_x, above), scene.goal])


def make_demos(
    waypoints: np.ndarray,  # (W, 2)
    n_demos: int,
    noise: float,  # std of the smooth offset, domain units
    rng: np.random.Generator,
    samples: int = 200,
) -> list[np.ndarray]:  # n_demos x (samples, 2)
    """The waypoint polyline plus smooth noise, clipped to 2.5 noise so a 0.06 gap holds it.

    The noise is white noise smoothed over a tenth of the path and rescaled to std noise; the
    ends stay where the waypoints put them.
    """
    base = resample(waypoints, samples)
    taper = np.sin(np.linspace(0.0, np.pi, samples))[:, None]
    demos = []
    for _ in range(n_demos):
        offset = gaussian_filter1d(rng.normal(size=(samples, 2)), samples / 10, axis=0)
        offset *= noise / offset.std()
        demos.append(base + taper * np.clip(offset, -2.5 * noise, 2.5 * noise))
    return demos


def master_path(
    demos: list[np.ndarray], n: int, sigma_min: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # s (n,), mu (n, 2), sigma (n,)
    """Demos aligned by normalised arc length and averaged; sigma(s) the isotropic spread, floored."""
    aligned = np.stack([resample(d, n) for d in demos])  # (D, n, 2)
    mean = aligned.mean(axis=0)
    spread = np.sqrt(((aligned - mean) ** 2).sum(axis=2).mean(axis=0) / 2)
    return np.linspace(0.0, 1.0, n), mean, np.maximum(spread, sigma_min)
