"""M0: every spatial-statistic variant on one static target, in an open domain and in Scene A.

Writes results/M0/: config.json, metrics.json, and per setting trajectories_<setting>.png and
metric_<setting>.png. Deterministic: there is no noise anywhere in M0.
"""

import json
import time
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from experiments.sequential_ergodic.basis import make_basis, mixture_coefficients
from experiments.sequential_ergodic.controller import Plant, simulate
from experiments.sequential_ergodic.scene_a import SceneA
from experiments.sequential_ergodic.statistic import (
    exponential,
    frequency_forgetting,
    full_memory,
    hard_reset,
)
from experiments.sequential_ergodic.walls import move

OUT = Path(__file__).parent / "results" / "M0"
CONFIG = {
    "K": [16, 16],
    "dt": 0.01,
    "u_max": 0.2,
    "T": 120.0,
    "start": [0.5, 0.15],
    "scene": {"gap_spacing": 0.25, "gap_width": 0.06},
    # Mass on both sides of Scene A's wall; above it, reached only through gap A.
    "target": {"means": [[0.25, 0.75], [0.65, 0.25]], "sigmas": [0.07, 0.07], "weights": [1, 1]},
    "hard_reset_threshold": 0.05,
    "exponential_lambda": 0.1,
    "frequency": {"lam_lo": 0.02, "lam_hi": 0.5, "kappa": 4.0},
    "pinned_speed_share": 0.05,  # a step shorter than this share of u_max dt counts as pinned
}
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4


def _variants(basis, cfg: dict) -> dict:
    f = cfg["frequency"]
    return {
        "full memory": full_memory(basis),
        f"hard reset (threshold {cfg['hard_reset_threshold']})": hard_reset(
            basis, cfg["hard_reset_threshold"]
        ),
        f"exponential, lambda = {cfg['exponential_lambda']}": exponential(
            basis, cfg["exponential_lambda"]
        ),
        f"frequency, lambda {f['lam_hi']} -> {f['lam_lo']}": frequency_forgetting(
            basis, f["lam_lo"], f["lam_hi"], f["kappa"]
        ),
    }


def _target_density(cfg: dict, grid: np.ndarray) -> np.ndarray:  # (G, G), indexed [y, x]
    X, Y = np.meshgrid(grid, grid)
    t = cfg["target"]
    d = sum(
        w * np.exp(-((X - m[0]) ** 2 + (Y - m[1]) ** 2) / (2 * s**2)) / s**2
        for m, s, w in zip(t["means"], t["sigmas"], t["weights"], strict=True)
    )
    return d / d.max()


def _draw_trajectories(runs: dict, walls: np.ndarray, cfg: dict, title: str) -> Figure:
    fig = Figure(figsize=(14, 4.1), layout="constrained")
    fig.suptitle(title)
    axes = fig.subplots(1, len(runs))
    grid = np.linspace(0, 1, 201)
    density = _target_density(cfg, grid)
    for ax, (name, (path, _, _)) in zip(axes, runs.items(), strict=True):
        ax.contourf(grid, grid, density, levels=np.linspace(0.1, 1, 8), cmap="Greys", alpha=0.5)
        segments = np.stack([path[:-1], path[1:]], axis=1)
        line = LineCollection(segments, cmap="viridis", lw=0.7)
        line.set_array(cfg["dt"] * np.arange(len(segments)))
        ax.add_collection(line)
        for a, b in walls:
            ax.plot(*np.array([a, b]).T, "-", color="0.2", lw=1.2)
        ax.plot(*cfg["start"], "o", color="k", ms=6)
        ax.plot(*path[-1], "s", mfc="none", mec="k", ms=8)
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", title=name)
    fig.colorbar(line, ax=axes, label="time (s)", shrink=0.8)
    return fig


def _draw_metric(runs: dict, cfg: dict, title: str) -> Figure:
    fig = Figure(figsize=(7, 4.5), layout="constrained")
    ax = fig.subplots()
    for colour, (name, (_, metric, _)) in zip(COLOURS, runs.items(), strict=True):
        t = cfg["dt"] * np.arange(1, len(metric) + 1)
        ax.loglog(t, metric, color=colour, lw=2, label=name)
    t_ref = np.array([1.0, cfg["T"]])
    m0 = next(iter(runs.values()))[1][99]  # full memory at t = 1 s
    ax.loglog(t_ref, m0 / t_ref, ":", color="0.55", lw=1.5, label="1/t")
    ax.loglog(t_ref, m0 / t_ref**2, "--", color="0.55", lw=1.5, label="1/t^2")
    ax.set(xlabel="time (s)", ylabel=r"ergodic metric $\sum_k \Lambda_k (\hat c_k - p_k)^2$")
    ax.set(xlim=(0.1, cfg["T"]), title=title)
    ax.grid(True, which="major", color="0.9", lw=0.6)
    ax.legend(frameon=False)
    return fig


def _summary_numbers(runs: dict, cfg: dict, scene: SceneA) -> dict:
    out = {}
    pinned = cfg["pinned_speed_share"] * cfg["u_max"] * cfg["dt"]
    for name, (path, metric, seconds) in runs.items():
        t = cfg["dt"] * np.arange(1, len(metric) + 1)
        late = t >= 5.0
        step = np.linalg.norm(np.diff(path, axis=0), axis=1)
        out[name] = {
            "final_metric": float(metric[-1]),
            "loglog_slope_t_ge_5s": float(np.polyfit(np.log(t[late]), np.log(metric[late]), 1)[0]),
            "final_position": path[-1].round(4).tolist(),
            "share_of_time_in_pocket_region": float(np.mean([scene.in_pocket(x) for x in path])),
            "share_of_steps_pinned": float((step < pinned).mean()),
            "wall_clock_s": seconds,
        }
    return out


def _run_setting(basis, p: np.ndarray, walls: np.ndarray, cfg: dict) -> dict:
    plant = Plant(lambda x, d: move(x, d, walls), cfg["dt"], cfg["u_max"])
    runs = {}
    for name, statistic in _variants(basis, cfg).items():
        start = time.perf_counter()
        path, metric = simulate(
            basis, statistic, p, np.array(cfg["start"]), plant, round(cfg["T"] / cfg["dt"])
        )
        runs[name] = (path, metric, time.perf_counter() - start)
    return runs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))
    basis = make_basis(tuple(cfg["K"]))
    scene = SceneA(**cfg["scene"])
    t = cfg["target"]
    p = mixture_coefficients(basis, np.array(t["means"]), np.array(t["sigmas"]), t["weights"])
    numbers = {}
    for setting, walls in (("open", np.empty((0, 2, 2))), ("scene_a", scene.walls())):
        runs = _run_setting(basis, p, walls, cfg)
        title = "open domain" if setting == "open" else "Scene A walls"
        fig = _draw_trajectories(runs, walls, cfg, f"Static target, {title}")
        fig.savefig(OUT / f"trajectories_{setting}.png", dpi=150)
        _draw_metric(runs, cfg, title).savefig(OUT / f"metric_{setting}.png", dpi=150)
        numbers[setting] = _summary_numbers(runs, cfg, scene)
    (OUT / "metrics.json").write_text(json.dumps(numbers, indent=2))
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
