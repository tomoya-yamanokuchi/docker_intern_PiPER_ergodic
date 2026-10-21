"""method/ against the 2-D notebook it was ported from, on the notebook's own data (d = 3).

The notebook's code cells are executed (its figures skipped), and every quantity the arm
runs on is compared with the notebook's own: the kernel width, the coefficients, the
labels, the sliding weights, the phase filter and the law's gradient. The notebook's
phase reads position only, so the filter and the labels are given its first two axes,
as the notebook gives them. The cube is not compared: the notebook's spans its world,
the arm's its data.

Run from final/src:  sh run.sh -c "import method.test_against_notebook as t; t.run()"
"""

import json

import matplotlib
import numpy as np
import tt

from method.law import Law, PhaseFilter, sliding_weights, sobolev_weights
from method.task import Task, coefficients, kernel_width, phase_label
from robot import SRC

NOTEBOOK = SRC / "notebooks/Sequential_E2T2_peg_slide_2D.ipynb"
SKIP = (
    "fig,",
    "OFFSET",
    "x_c,",
    "for offset",
    "w_chk",
    "walk",
    "world =",
    "for i in range(d)",
    "def width_sampled",
)
nb: dict = {}


def load() -> dict:
    if not nb:
        matplotlib.use("Agg")
        for cell in json.loads(NOTEBOOK.read_text())["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "code" and not source.startswith(SKIP):
                exec(source, nb)  # the notebook is ours and trusted
    return nb


def notebook_task() -> Task:
    n = load()
    return Task(
        np.eye(3), n["master"], np.zeros((len(n["master"]), 6)), n["s_grid"], n["X_p"],
        np.zeros((len(n["X_p"]), 6)), n["phi_p"], n["CUBE_LO"], n["L"], n["h"],
    )  # fmt: skip


def test_kernel_width_matches() -> None:
    n = load()
    assert abs(kernel_width(n["X_p"], n["CUBE_LO"], n["L"]) - n["h"]) < 1e-9 * n["h"]


def test_coefficients_match() -> None:
    n = load()
    for mine, theirs in zip(
        coefficients(n["X_p"], n["h"], n["CUBE_LO"], n["L"]), n["C"], strict=True
    ):
        assert np.abs(mine - theirs).max() < 1e-14


def test_resolvable_width_matches() -> None:
    assert abs(notebook_task().sigma_f - load()["sigma_f"]) < 1e-15


def test_labels_match() -> None:
    n = load()
    X = n["states"][::7]
    assert np.array_equal(
        phase_label(X[:, :2], n["master"][:, :2], n["phi_master"]), n["phase_label"](X)
    )


def test_sliding_weights_match() -> None:
    n = load()
    for phi in (0.0, 0.15, 0.35, 0.8, 1.0):
        for rho in (0.0, 0.05, 0.3):
            assert (
                np.abs(
                    sliding_weights(phi, n["phi_p"], n["delta"], rho) - n["weights"](phi, rho)
                ).max()
                < 1e-15
            )


def test_phase_filter_matches() -> None:
    n = load()
    path = n["replay"](
        n["Slide"](), np.array([n["MASTER"][0], n["MASTER"][1], n["MASTER"][0]]), 3000
    )
    mine, theirs = PhaseFilter(n["master"][:, :2], n["s_grid"], n["h"]), n["PhaseFilter"]()
    assert max(abs(mine.update(x[:2]) - theirs.update(x)) for x in path) < 1e-12


def test_gradient_matches() -> None:
    """The law's contraction against the notebook's Lambda * (W - W_hat), on its own statistic."""
    n = load()
    law = Law(notebook_task(), "retry", n["DT"])
    law.lam_cores = tt.vector.to_list(n["ttLambda"])
    law.stat = n["Statistic"](1.0)
    rng = np.random.default_rng(0)
    for x in n["states"][rng.choice(len(n["states"]), 30)]:
        law.stat.add(n["tt_phi_dphi"](x)[0])
    for x in n["states"][rng.choice(len(n["states"]), 10)]:
        w = n["weights"](rng.uniform(), rng.uniform(0, 0.2))
        dvW = n["ttLambda"] * (law.stat.W() - n["target"](w))
        theirs = np.array([tt.dot(dvW, dphi) for dphi in n["tt_phi_dphi"](x)[1]])
        mine = law.gradient(*_basis(n, x), w)
        assert np.abs(mine - theirs).max() < 1e-9 * np.abs(theirs).max()


def _basis(n: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.array([n["Phi"](xi, i) for i, xi in enumerate(x)]), np.array(
        [n["DPhi"](xi, i) for i, xi in enumerate(x)]
    )


def test_sobolev_weights_match_dense_6d() -> None:
    """Lambda's TT against the formula, on random 6-D indices, to the cross's own accuracy."""
    d, K = 6, 10
    idx = np.random.default_rng(0).integers(0, K, (2000, d))
    dense = (1.0 + np.linalg.norm(idx, axis=1) ** 2) ** (-(d + 1) / 2)
    cores = tt.vector.to_list(sobolev_weights(d))
    approx = np.array(
        [
            np.linalg.multi_dot([c[:, i, :] for c, i in zip(cores, k, strict=True)]).item()
            for k in idx
        ]
    )
    assert np.abs(approx - dense).max() < 2e-2 * dense.max()


def run() -> None:
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
