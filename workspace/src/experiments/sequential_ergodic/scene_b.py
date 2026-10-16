"""Scene B: the peg_slide environment in the controller's cube, z = s / SIZE.

The demonstrations follow the master clicked in the peg_slide notebook: down into the slot, then
along, pushing the block to the goal.
"""

from pathlib import Path

import numpy as np

from simulation.peg_slide import SIZE, Slide, missed, slide_step, succeeded

TEACH_FILE = (
    Path(__file__).parents[3]
    / "saved_recordings/exploration/peg_slide_2D/peg_slide_2D_teaching.npz"
)


def master_waypoints() -> np.ndarray:  # (W, 2) cube
    return np.load(TEACH_FILE)["master"] / SIZE


def move(world: Slide, z: np.ndarray, d: np.ndarray) -> np.ndarray:  # (2,) cube
    return slide_step(world, z * SIZE, (z + d) * SIZE) / SIZE


def done(world: Slide, z: np.ndarray) -> bool:
    return succeeded(world, z * SIZE)


def in_dead_end(world: Slide, z: np.ndarray) -> bool:
    return missed(world, z * SIZE)


def follow(
    world: Slide, waypoints: np.ndarray, step: float, stuck_ticks: int = 25
) -> tuple[np.ndarray, np.ndarray]:  # (T, 2) cube, (T,) block centre
    """A scripted peg, moved through the environment towards each world waypoint in turn.

    It moves `step` world units a tick and takes the next waypoint once within a step of this one, or
    once it has not moved for stuck_ticks ticks (a face stopped it). The world changes as it goes.
    """
    s = np.array(waypoints[0], dtype=float)
    path, cs = [s], [world.c]
    for goal in waypoints[1:]:
        still = 0
        while np.hypot(*(goal - s)) > step and still < stuck_ticks:
            d = goal - s
            s_new = slide_step(world, s, s + d * min(1.0, step / np.hypot(*d)))
            still = still + 1 if np.allclose(s_new, s) else 0
            s = s_new
            path.append(s)
            cs.append(world.c)
    return np.array(path) / SIZE, np.array(cs)


def slider_phase_cap(s: np.ndarray, mu: np.ndarray, pushed: float, lead: float) -> float:
    """The master's phase once the block has been pushed `pushed` world units, plus lead.

    The push leg starts where the master first comes within 0.01 (cube) of its lowest point, the
    slot's floor; along it the peg's x advances as the block does.
    """
    bottom = int(np.argmax(mu[:, 1] <= mu[:, 1].min() + 0.01))
    along = np.maximum.accumulate(mu[bottom:, 0] - mu[bottom, 0])
    i = bottom + int(np.searchsorted(along, pushed / SIZE))
    return float(min(1.0, s[min(i, len(s) - 1)] + lead))
