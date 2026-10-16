import numpy as np

from experiments.sequential_ergodic.basis import make_basis, values
from experiments.sequential_ergodic.demos import resample
from experiments.sequential_ergodic.detector import Firing, NegativeProgressDetector
from experiments.sequential_ergodic.marks import (
    MarkingTarget,
    MarkParams,
    build_mark,
    separate_terms,
    zeroing_scale,
)
from experiments.sequential_ergodic.phase import PhaseEstimator
from experiments.sequential_ergodic.statistic import full_memory
from experiments.sequential_ergodic.target import ExpandingTarget, PhaseConditionedTarget

BASIS = make_basis((8, 8))
DT = 0.01
TIMES = DT * np.arange(400)
PATH = np.column_stack([np.linspace(0.2, 0.8, 400), np.full(400, 0.4)])
FIRING = Firing(t_peak=1.0, t_det=3.0, phi_peak=0.6, phi_det=0.5, branch="cusum")
PARAMS = MarkParams(gamma=0.0)


def test_mark_shape_is_the_ramped_average_of_the_basis_over_the_segment():
    mark = build_mark(BASIS, TIMES, PATH, FIRING, PARAMS)
    seg = (TIMES >= 1.0) & (TIMES <= 3.0)
    w = np.minimum(1.0, (TIMES[seg] - 1.0) / PARAMS.tau_ramp)
    expected = sum(wi * values(BASIS, x) for wi, x in zip(w, PATH[seg], strict=True)) / w.sum()
    assert np.allclose(mark.g, expected)
    assert np.isclose(mark.mass, 2.0) and mark.phi == 0.5 and np.isclose(mark.sigma, 0.1)
    at_peak = build_mark(BASIS, TIMES, PATH, FIRING, MarkParams(tag_at_peak=True))
    assert at_peak.phi == 0.6


def test_separate_marks_weigh_by_phase_and_global_ones_do_not():
    local = build_mark(BASIS, TIMES, PATH, FIRING, PARAMS)
    glob = build_mark(BASIS, TIMES, PATH, FIRING, PARAMS, global_=True)
    t = 3.0 + PARAMS.T_ramp  # fully ramped in
    _, at_mark = separate_terms([local], 0.5, t, PARAMS)
    _, far = separate_terms([local], 0.9, t, PARAMS)
    assert np.isclose(at_mark, 2.0) and far < 1e-3
    assert np.isclose(separate_terms([glob], 0.9, t, PARAMS)[1], 2.0)


def test_zeroing_removes_components_on_the_segment_only():
    mark = build_mark(BASIS, TIMES, PATH, FIRING, PARAMS)
    means = np.array([[0.4, 0.4], [0.4, 0.9]])
    scale = zeroing_scale([mark], means, np.array([0.03, 0.03]), 3.0 + PARAMS.T_ramp, PARAMS)
    assert scale[0] < 1e-6 and scale[1] > 0.99


def _marking(variant: str) -> MarkingTarget:
    s = np.linspace(0, 1, 51)
    mu = resample(np.array([[0.1, 0.1], [0.9, 0.9]]), 51)
    sigma = np.full(51, 0.03)
    est = PhaseEstimator(s, mu, sigma)
    est.reset(mu[0])
    inner = PhaseConditionedTarget(est, ExpandingTarget(BASIS, s, mu, sigma), DT)
    target = MarkingTarget(
        inner, full_memory(BASIS), BASIS, NegativeProgressDetector(), variant, PARAMS
    )
    target.marks.append(build_mark(BASIS, TIMES, PATH, FIRING, PARAMS))
    return target


def test_inside_marks_add_their_mass_to_the_statistic_over_the_ramp():
    target = _marking("inside")
    mark = target.marks[0]
    for n in range(round(PARAMS.T_ramp / DT) + 50):
        target._apply(np.zeros(len(BASIS.k)), mark.t_fire + n * DT, DT)
    assert np.allclose(target.statistic.Z, mark.mass) and np.allclose(
        target.statistic.A, mark.mass * mark.g
    )


def test_negative_marks_subtract_their_shape_on_the_statistics_scale():
    target = _marking("negative")
    target.statistic.Z = np.full(len(BASIS.k), 4.0)
    mark, p = target.marks[0], np.ones(len(BASIS.k))
    p_eff = target._apply(p, mark.t_fire + PARAMS.T_ramp, DT)
    assert np.allclose(p_eff, p - PARAMS.w_neg * mark.mass * mark.g / 4.0)
