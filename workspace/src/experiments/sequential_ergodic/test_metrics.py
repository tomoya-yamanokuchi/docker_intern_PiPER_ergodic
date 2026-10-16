import numpy as np

from experiments.sequential_ergodic.metrics import bootstrap_ci, revisit_overlap, visits


def test_visits_are_the_runs_of_true():
    inside = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert visits(inside) == [slice(0, 2), slice(4, 5), slice(6, 9)]


def test_revisit_overlap_is_the_share_of_samples_in_cells_seen_last_visit():
    path = np.array([[0.01, 0.01], [0.03, 0.01], [0.5, 0.5], [0.01, 0.015], [0.07, 0.01]])
    # visit 2 = samples 3 and 4: sample 3 is in visit 1's first cell, sample 4 in none of them
    assert revisit_overlap(path, [slice(0, 2), slice(3, 5)], cell=0.02) == [0.5]


def test_bootstrap_ci_matches_the_normal_interval_for_normal_data():
    rng = np.random.default_rng(0)
    v = rng.normal(size=400)
    mean, lo, hi = bootstrap_ci(v, rng)
    half = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    assert np.isclose(hi - lo, 2 * half, rtol=0.15) and lo < mean < hi
