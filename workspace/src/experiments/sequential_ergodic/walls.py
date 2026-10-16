"""Walls as line segments: a step that would cross one stops at it and slides along it."""

import numpy as np

MARGIN = 1e-6  # how far off a wall a stopped agent is left, so that it never lies on one
SLIDES = 3  # a step, then up to two slides along the walls it meets


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def first_hit(x: np.ndarray, d: np.ndarray, walls: np.ndarray) -> tuple[float, int] | None:
    """(t, wall index) of the first wall (W, 2, 2) the segment x -> x + d meets, 0 <= t <= 1."""
    best = None
    for i, (a, b) in enumerate(walls):
        r = b - a
        denom = _cross(d, r)
        if abs(denom) < 1e-15:  # parallel: sliding along a wall never crosses it
            continue
        t, s = _cross(a - x, r) / denom, _cross(a - x, d) / denom
        if 0.0 <= t <= 1.0 and 0.0 <= s <= 1.0 and (best is None or t < best[0]):
            best = (t, i)
    return best


def move(x: np.ndarray, d: np.ndarray, walls: np.ndarray) -> np.ndarray:  # (2,)
    """Where the agent at x (2,) ends up when commanded the step d (2,), inside [0, 1]^2.

    The step is clipped to the domain first, so it cannot go round the end of a wall that
    meets the domain's edge.
    """
    x = np.asarray(x, dtype=float)
    d = np.clip(x + d, MARGIN, 1.0 - MARGIN) - x
    for _ in range(SLIDES):
        hit = first_hit(x, d, walls)
        if hit is None:
            return x + d
        t, i = hit
        a, b = walls[i]
        r = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-r[1], r[0]])
        side = np.sign(normal @ (x - a))
        contact = x + t * d
        x = contact + side * MARGIN * normal
        d = ((1.0 - t) * d @ r) * r
    return x
