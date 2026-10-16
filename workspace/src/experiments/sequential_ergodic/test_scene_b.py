import numpy as np

from experiments.sequential_ergodic import scene_b
from experiments.sequential_ergodic.demos import resample
from simulation.peg_slide import BLOCK_TOP, C0, SIZE, TRAVEL, Slide, slide_step


def test_move_is_slide_step_in_the_cube():
    z, d = np.array([0.2, 0.35]), np.array([0.01, -0.02])
    expected = slide_step(Slide(), z * SIZE, (z + d) * SIZE) / SIZE
    assert np.allclose(scene_b.move(Slide(), z, d), expected)


def test_follow_reaches_free_waypoints_and_slides_over_the_block():
    world = Slide()
    path, cs = scene_b.follow(world, np.array([[1.0, 5.0], [3.5, 5.0], [3.5, 3.3]]), 0.02)
    assert np.allclose(path[-1] * SIZE, [3.5, 3.3], atol=0.02) and np.all(cs == C0)
    assert np.all(path[:, 1] * SIZE > BLOCK_TOP)


def test_slider_cap_runs_from_the_slot_floor_to_the_end_of_the_push():
    s = np.linspace(0, 1, 101)
    mu = resample(np.array([[2.3, 6.9], [2.3, 1.6], [8.8, 1.6]]) / SIZE, 101)
    bottom = s[int(np.argmax(mu[:, 1] <= mu[:, 1].min() + 0.01))]
    assert np.isclose(scene_b.slider_phase_cap(s, mu, 0.0, 0.05), bottom + 0.05)
    assert scene_b.slider_phase_cap(s, mu, TRAVEL, 0.05) == 1.0
    halfway = scene_b.slider_phase_cap(s, mu, TRAVEL / 2, 0.0)
    assert bottom < halfway < 1.0
