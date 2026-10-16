"""The 2-D peg-slide environment of Ergodic_Exploration_peg_slide_2D.ipynb, in world units.

A 10 x 10 world; the agent is the peg tip, a point with the peg's half-width taken off every gap.
The block (centre c) has a slot open at the top; a barrier at x = 4 stops the tip and the block until
the tip has reached the slot's floor; a tip in the slot pushes the block along, as far as the goal.
"""

from dataclasses import dataclass

import numpy as np

SIZE = 10.0  # the world's side, and world units per cube unit
PEG_HALF, SLOT_HALF, BLOCK_HALF = 0.1, 0.3, 0.75
CLEARANCE = SLOT_HALF - PEG_HALF  # the tip's play to either side of the slot's centre
BLOCK_BOTTOM, BLOCK_TOP = 1.0, 3.0
SLOT_FLOOR = 1.5
INSERTED = 0.15  # the tip this close to the slot's floor removes the barrier
BARRIER_X, BARRIER_Y = 4.0, (0.5, 8.0)
C0 = 2.3  # the block's centre as taught
TRAVEL = 6.0  # from there to the goal
GOAL_C = C0 + TRAVEL  # the block's centre at the goal, wherever it started
GOAL_X = GOAL_C - 0.1  # the peg and the block's centre both beyond this is the task done


@dataclass
class Slide:
    home: float = C0  # the block's centre at the start
    c: float | None = None  # the block's centre now
    barrier: bool = True

    def __post_init__(self) -> None:
        if self.c is None:
            self.c = self.home


def in_slot(c: float, x: float, y: float) -> bool:
    # The margin keeps a tip that was clamped to a slot wall inside, whatever the rounding.
    return abs(x - c) <= CLEARANCE + 1e-9 and SLOT_FLOOR <= y < BLOCK_TOP


def free(world: Slide, s: np.ndarray, x: float, y: float) -> bool:
    """Whether the tip can move from s (2,) to (x, y): not into the block, not through the barrier."""
    in_block = abs(x - world.c) < BLOCK_HALF + PEG_HALF and BLOCK_BOTTOM < y < BLOCK_TOP
    crosses = (s[0] - BARRIER_X) * (x - BARRIER_X) < 0 and BARRIER_Y[0] <= y <= BARRIER_Y[1]
    return (not in_block or in_slot(world.c, x, y)) and not (world.barrier and crosses)


def slide_step(
    world: Slide,
    s: np.ndarray,  # (2,) the tip now
    s_new: np.ndarray,  # (2,) the tip commanded
) -> np.ndarray:  # (2,)
    """The tip the environment lets the robot reach from s; moves the block if it is pushed."""
    nx, ny = np.clip(s_new, 0.0, SIZE)
    if not (in_slot(world.c, *s) and ny < BLOCK_TOP):
        # Outside the slot, or leaving it through the top: stopped faces let the tip slide along them.
        for x, y in ((nx, ny), (nx, s[1]), (s[0], ny)):
            if free(world, s, x, y):
                return np.array([x, y])
        return np.array(s)
    ny = max(ny, SLOT_FLOOR)
    if ny <= SLOT_FLOOR + INSERTED:
        world.barrier = False
    stop = min(GOAL_C, BARRIER_X - BLOCK_HALF) if world.barrier else GOAL_C
    pushed = nx - np.clip(nx, world.c - CLEARANCE, world.c + CLEARANCE)
    world.c = float(np.clip(world.c + pushed, world.home, max(stop, world.c)))
    return np.array([np.clip(nx, world.c - CLEARANCE, world.c + CLEARANCE), ny])


def succeeded(world: Slide, s: np.ndarray) -> bool:
    """The task done: the peg and the block's centre both at the goal."""
    return bool(s[0] > GOAL_X and world.c > GOAL_X)


def missed(world: Slide, s: np.ndarray) -> bool:
    """The peg below the block's top beside the block, on either side, while the barrier stands."""
    beside = abs(s[0] - world.c) > BLOCK_HALF + PEG_HALF and s[0] < BARRIER_X
    return bool(world.barrier and s[1] < BLOCK_TOP and beside)
