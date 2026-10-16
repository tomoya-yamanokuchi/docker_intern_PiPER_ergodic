"""Episode metrics and their confidence intervals."""

from itertools import pairwise

import numpy as np


def visits(inside: np.ndarray) -> list[slice]:
    """The maximal runs of True in inside (T,), as slices of it."""
    edges = np.diff(np.concatenate([[0], inside.astype(int), [0]]))
    return [
        slice(a, b)
        for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0], strict=True)
    ]


def _cells(points: np.ndarray, cell: float) -> np.ndarray:  # (P, 2) -> (P,) integer cell ids
    ij = np.floor(points / cell).astype(int)
    return ij[:, 0] * 100_000 + ij[:, 1]


def revisit_overlap(path: np.ndarray, runs: list[slice], cell: float) -> list[float]:
    """Per visit after the first: the share of its samples in cells the previous visit touched."""
    out = []
    for previous, current in pairwise(runs):
        seen = set(_cells(path[previous], cell).tolist())
        out.append(float(np.mean([c in seen for c in _cells(path[current], cell)])))
    return out


def bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, n: int = 2000
) -> tuple[float, float, float]:
    """(mean, lower, upper) of a 95 % percentile bootstrap; nan for no values."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    means = rng.choice(v, size=(n, len(v))).mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
