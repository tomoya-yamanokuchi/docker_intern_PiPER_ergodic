"""The recall law: a target over the whole past, the statistic's forgetting pulls the law back.

Sliding's target holds only the datapoints from the phase forward. Here the forward edge is
the same, but nothing behind the phase is removed: a datapoint a phase-distance a behind it
weighs 2^-a, halving over a full master length as the statistic does over one forgetting
horizon, so no constant of its own. Going back in phase takes the datapoints ahead of the
phase out of the target. There is no stall timer and no retreating edge; a stage the
statistic has forgotten is a stage the law is drawn to again.
"""

import numpy as np

from method.law import Law


def recall_weights(phi: float, phi_p: np.ndarray, delta: float) -> np.ndarray:
    """(N_P,) of sum 1: the datapoints up to 2 delta ahead of the phase, the past halving per master length."""
    w = 1.0 / (1.0 + np.exp(-(phi + 2 * delta - phi_p) / delta))
    w = w * 2.0 ** -np.maximum(phi - phi_p, 0.0)
    return w / w.sum()


class RecallLaw(Law):
    def weights(self, phi: float) -> np.ndarray:
        return recall_weights(phi, self.task.phi_p, self.delta)
