import numpy as np

from experiments.sequential_ergodic.basis import make_basis, mixture_coefficients
from experiments.sequential_ergodic.controller import Plant, simulate
from experiments.sequential_ergodic.scene_a import SceneA
from experiments.sequential_ergodic.statistic import full_memory
from experiments.sequential_ergodic.walls import MARGIN, move

SCENE = SceneA()
WALLS = SCENE.walls()


def _crossings(path: np.ndarray, scene: SceneA) -> int:
    """Steps of path (T, 2) that cross a wall, by the scene's geometry rather than the segments.

    Each step is short against the walls' spacing, so a crossing is a change of side of a wall's
    line with the crossing point inside the wall's extent.
    """
    a, b = path[:-1], path[1:]
    count = 0
    for axis, level, lo, hi, solid in (
        (1, scene.wall_y, 0.0, 1.0, _main_wall_solid(scene)),
        (0, scene.gap_b_x - scene.pocket_half, scene.wall_y, scene.pocket_top, None),
        (0, scene.gap_b_x + scene.pocket_half, scene.wall_y, scene.pocket_top, None),
        (
            1,
            scene.pocket_top,
            scene.gap_b_x - scene.pocket_half,
            scene.gap_b_x + scene.pocket_half,
            None,
        ),
    ):
        flips = (a[:, axis] - level) * (b[:, axis] - level) < 0
        t = (level - a[:, axis]) / np.where(flips, b[:, axis] - a[:, axis], 1.0)
        along = a[:, 1 - axis] + t * (b[:, 1 - axis] - a[:, 1 - axis])
        hit = flips & (along >= lo) & (along <= hi)
        if solid is not None:
            hit &= solid(along)
        count += int(hit.sum())
    return count


def _main_wall_solid(scene: SceneA):
    g = scene.gap_width / 2
    return lambda x: (np.abs(x - scene.gap_a_x) > g) & (np.abs(x - scene.gap_b_x) > g)


def test_a_step_into_a_wall_stops_short_and_slides_along_it():
    x = move(np.array([0.1, 0.49]), np.array([0.02, 0.02]), WALLS)
    assert np.allclose(x, [0.12, SCENE.wall_y - MARGIN], rtol=0, atol=1e-9)


def test_a_step_through_a_gap_passes():
    x = move(np.array([SCENE.gap_a_x, 0.49]), np.array([0.0, 0.02]), WALLS)
    assert np.allclose(x, [SCENE.gap_a_x, 0.51])


def test_random_steps_never_cross_a_wall():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(20000, 2))
    d = rng.normal(scale=0.03, size=(20000, 2))
    moved = np.array([move(xi, di, WALLS) for xi, di in zip(x, d, strict=True)])
    paths = np.stack([x, moved], axis=1)
    assert sum(_crossings(p, SCENE) for p in paths) == 0


def test_the_agent_never_crosses_a_wall():
    basis = make_basis((16, 16))
    # Mass in the pocket, beyond gap A and below the wall: the agent is drawn at every wall.
    means = np.array([[SCENE.gap_b_x, 0.6], [0.2, 0.8], [0.5, 0.2]])
    p = mixture_coefficients(basis, means, np.full(3, 0.05), np.ones(3))
    path, _ = simulate(
        basis,
        full_memory(basis),
        p,
        np.array(SCENE.start),
        Plant(lambda x, d: move(x, d, WALLS)),
        6000,
    )
    assert _crossings(path, SCENE) == 0
    assert np.all((path >= 0.0) & (path <= 1.0))
    assert any(SCENE.in_pocket(x) for x in path)  # it did get through a gap
