"""Scene A: a horizontal wall with two gaps. Gap A leads on to the goal, gap B into a closed pocket.

The start lies below the wall nearer gap B than gap A, so the agent tends to try B first.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SceneA:
    gap_spacing: float = 0.25  # between the two gaps' centres
    gap_width: float = 0.06
    wall_y: float = 0.5
    gap_b_x: float = 0.55  # gap A lies gap_spacing to its left
    pocket_top: float = 0.7
    start: tuple[float, float] = (0.5, 0.15)
    goal: tuple[float, float] = (0.15, 0.85)
    goal_radius: float = 0.05

    def __post_init__(self) -> None:
        # The pocket's left wall must fit between the two gaps.
        assert self.gap_spacing > self.gap_width, "gaps closer than their width overlap"

    @property
    def gap_a_x(self) -> float:
        return self.gap_b_x - self.gap_spacing

    @property
    def pocket_half(self) -> float:
        """The pocket's half-width: 0.1, or less when the gaps are too close for it."""
        return min(0.1, self.gap_spacing / 2)

    def walls(self) -> np.ndarray:  # (W, 2, 2)
        y, g, xa, xb, ph = (
            self.wall_y,
            self.gap_width / 2,
            self.gap_a_x,
            self.gap_b_x,
            self.pocket_half,
        )
        top = self.pocket_top
        segments = [
            ((0.0, y), (xa - g, y)),
            ((xa + g, y), (xb - g, y)),
            ((xb + g, y), (1.0, y)),
            ((xb - ph, y), (xb - ph, top)),
            ((xb + ph, y), (xb + ph, top)),
            ((xb - ph, top), (xb + ph, top)),
        ]
        return np.array(segments, dtype=float)

    def reached_goal(self, x: np.ndarray) -> bool:
        return bool(np.hypot(*(np.asarray(x) - self.goal)) <= self.goal_radius)

    def in_pocket(self, x: np.ndarray) -> bool:
        return bool(
            abs(x[0] - self.gap_b_x) < self.pocket_half and self.wall_y < x[1] < self.pocket_top
        )
