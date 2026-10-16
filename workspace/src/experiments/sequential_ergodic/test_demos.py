from itertools import pairwise

import numpy as np

from experiments.sequential_ergodic.demos import (
    make_demos,
    master_path,
    resample,
    scene_a_waypoints,
)
from experiments.sequential_ergodic.scene_a import SceneA
from experiments.sequential_ergodic.walls import first_hit

SCENE = SceneA()


def test_resampling_spaces_points_equally_in_arc_length():
    path = np.array([[0.0, 0.0], [0.3, 0.0], [0.3, 0.1]])
    steps = np.linalg.norm(np.diff(resample(path, 41), axis=0), axis=1)
    assert np.allclose(steps, 0.4 / 40)


def test_demos_never_cross_a_wall():
    for seed in range(20):
        for demo in make_demos(scene_a_waypoints(SCENE), 8, 0.008, np.random.default_rng(seed)):
            for a, b in pairwise(demo):
                assert first_hit(a, b - a, SCENE.walls()) is None


def test_master_of_one_path_at_different_speeds_is_that_path():
    line = np.array([[0.1, 0.1], [0.6, 0.1], [0.6, 0.9]])
    # Dense on the first leg, sparse on the second, the corner sampled by both.
    slow_start = np.vstack([resample(line[:2], 200), resample(line[1:], 9)[1:]])
    s, mu, sigma = master_path([line, slow_start], 101, 0.03)
    assert np.allclose(mu, resample(line, 101), atol=1e-9)
    assert np.allclose(sigma, 0.03) and s[0] == 0.0 and s[-1] == 1.0


def test_master_runs_from_the_start_to_the_goal():
    demos = make_demos(scene_a_waypoints(SCENE), 8, 0.008, np.random.default_rng(0))
    _, mu, _ = master_path(demos, 101, 0.03)
    assert np.allclose(mu[0], SCENE.start) and np.allclose(mu[-1], SCENE.goal)
