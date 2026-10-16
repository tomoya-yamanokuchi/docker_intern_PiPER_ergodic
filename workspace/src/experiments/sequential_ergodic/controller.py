"""The SMC law, shared by every condition: only the statistic and the target it reads change."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import islice

import numpy as np

from experiments.sequential_ergodic.basis import Basis, ergodic_metric, gradients, values
from experiments.sequential_ergodic.statistic import Statistic


@dataclass(frozen=True)
class Plant:
    """The first-order agent x' = u, |u| = u_max, in an environment.

    move(x, d) is where the agent at x (2,) ends up when commanded the step d (2,): walls,
    contact and anything the environment does about them.
    """

    move: Callable[[np.ndarray, np.ndarray], np.ndarray]
    dt: float = 0.01  # s
    u_max: float = 0.2  # domain units / s


def direction(basis: Basis, x: np.ndarray, e: np.ndarray, u_max: float) -> np.ndarray:  # (2,)
    """u = -u_max B / |B|, B = sum_k Lambda_k e_k grad F_k(x), e = c_hat - p."""
    B = (basis.lam * e) @ gradients(basis, x)
    return -u_max * B / max(float(np.linalg.norm(B)), 1e-12)


def trajectory(
    basis: Basis,
    statistic: Statistic,
    target: Callable[[np.ndarray], np.ndarray],  # x (2,) -> target coefficients p (N,)
    x0: np.ndarray,  # (2,)
    plant: Plant,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per step (x, c_hat, p): the target is read at x and x added to the statistic, then the
    agent moves by the law. Endless; the caller stops it."""
    x = np.asarray(x0, dtype=float)
    while True:
        p = target(x)
        statistic.update(values(basis, x), plant.dt, p)
        c = statistic.coefficients()
        yield x, c, p
        x = plant.move(x, plant.dt * direction(basis, x, c - p, plant.u_max))


def simulate(
    basis: Basis,
    statistic: Statistic,
    p: np.ndarray,  # (N,) static target coefficients
    x0: np.ndarray,  # (2,)
    plant: Plant,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:  # (steps + 1, 2) path, (steps,) ergodic metric
    path, metric = [], []
    for x, c, _ in islice(trajectory(basis, statistic, lambda _: p, x0, plant), steps + 1):
        path.append(x)
        metric.append(ergodic_metric(basis, c, p))
    return np.array(path), np.array(metric[:-1])
