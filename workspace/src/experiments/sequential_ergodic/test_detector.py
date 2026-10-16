import numpy as np

from experiments.sequential_ergodic.detector import NegativeProgressDetector

PERIOD = 0.1


def _run(phi: np.ndarray, speed: float = 0.2, **params) -> list:
    det = NegativeProgressDetector(**params)
    return [
        f for n, p in enumerate(phi) if (f := det.step(n * PERIOD, float(p), speed)) is not None
    ]


def test_steady_progress_never_fires():
    assert _run(np.linspace(0.0, 1.0, 200)) == []


def test_a_sustained_drop_fires_once_with_the_peak_where_it_was():
    phi = np.concatenate([np.linspace(0, 0.5, 51), np.linspace(0.5, 0.3, 21)])  # peak at step 50
    firings = _run(phi, h=0.02, delta_drop=0.03, T_ref=5.0)
    assert len(firings) == 1 and firings[0].branch == "cusum"
    assert np.isclose(firings[0].t_peak, 5.0) and firings[0].phi_peak == 0.5
    assert 5.0 < firings[0].t_det <= 6.0  # within a second of the turn


def test_a_flat_phase_while_moving_fires_the_stall_branch():
    firings = _run(np.full(40, 0.4), speed=0.2, T_stall=2.0)
    assert [f.branch for f in firings] == ["stall"] and np.isclose(firings[0].t_det, 2.0)


def test_standing_still_is_not_a_stall():
    assert _run(np.full(40, 0.4), speed=0.0) == []


def test_firings_are_at_least_T_ref_apart():
    phi = np.concatenate([np.linspace(0, 0.5, 51), np.linspace(0.5, 0.0, 200)])
    t = [f.t_det for f in _run(phi, K_c=0.0, h=0.02, delta_drop=0.03, T_ref=3.0)]
    assert len(t) >= 2 and np.all(np.diff(t) > 3.0)
