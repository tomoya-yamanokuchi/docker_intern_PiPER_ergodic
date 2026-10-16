"""Tables and figures of Scene B episodes, shared by the milestone scripts."""

import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from experiments.sequential_ergodic.episode import Job, run_episode, setup
from experiments.sequential_ergodic.metrics import bootstrap_ci, visits
from simulation.peg_slide import (
    BARRIER_X,
    BARRIER_Y,
    BLOCK_BOTTOM,
    BLOCK_HALF,
    BLOCK_TOP,
    GOAL_C,
    SIZE,
    SLOT_FLOOR,
    SLOT_HALF,
)

# Categorical slots 1-8, in fixed order; a figure takes them in its conditions' order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7b5cd6", "#8a6d3b"]


def run_all(jobs: list[Job], fn: Callable[[Job], dict] = run_episode) -> list[dict]:
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("fork")) as pool:
        return list(pool.map(fn, jobs, chunksize=2))


def row(eps: list[dict], rng: np.random.Generator) -> dict:
    """Mean and 95 % bootstrap CI of each metric over episodes; None values are left out."""

    def ci(values: list) -> list[float]:
        return list(bootstrap_ci(np.array([v for v in values if v is not None], dtype=float), rng))

    entered = [e for e in eps if e["entries"]]
    return {
        "n": len(eps),
        "success_rate": ci([e["success"] for e in eps]),
        "time_to_success": ci([e["time_to_success"] for e in eps]),
        "entered_dead_end": ci([bool(e["entries"]) for e in eps]),
        "entries": ci([e["entries"] for e in eps]),
        "time_in_dead_end": ci([e["time_in_dead_end"] for e in eps]),
        "first_dwell": ci([e["dwells"][0] for e in entered]),
        "dead_end_after_first_mark": ci([e["dead_end_after_first_mark"] for e in eps]),
        "entries_after_first_mark": ci([e["entries_after_first_mark"] for e in eps]),
        "n_marks": ci([e["n_marks"] for e in eps]),
        "mark_mass": ci([e["mark_mass"] for e in eps]),
        "marks_at_hole": ci([e["marks_at_hole"] for e in eps]),
        "false_mark_rate": ci([e["false_mark_rate"] for e in eps]),
        "dead_end_mark_share": ci([e["dead_end_mark_share"] for e in eps]),
        "final_metric": ci([e["final_metric"] for e in eps]),
        "revisit_overlap": ci([o for e in eps for o in e["overlaps"]]),
    }


TABLE_COLUMNS = {
    "success_rate": ("success", ".2f"),
    "time_to_success": ("time to success (s)", ".1f"),
    "entries": ("dead-end entries", ".1f"),
    "dead_end_after_first_mark": ("dead-end time after 1st mark (s)", ".1f"),
    "n_marks": ("marks", ".1f"),
    "false_mark_rate": ("false-mark rate", ".2f"),
    "final_metric": ("final metric", ".3f"),
}


def markdown_table(rows: dict[str, dict], columns: dict = TABLE_COLUMNS) -> str:
    def cell(v: list[float], form: str) -> str:
        return "—" if np.isnan(v[0]) else f"{v[0]:{form}} [{v[1]:{form}}, {v[2]:{form}}]"

    head = "| condition | " + " | ".join(label for label, _ in columns.values()) + " |"
    lines = [head, "|" + "---|" * (len(columns) + 1)]
    for name, r in rows.items():
        lines.append(
            f"| {name} | " + " | ".join(cell(r[k], f) for k, (_, f) in columns.items()) + " |"
        )
    return "\n".join(lines)


def draw_world(ax, c: float, barrier: bool, mu: np.ndarray | None = None) -> None:
    """The block at c with its slot, the barrier while it stands, the goal and the master, in world units."""
    for lo, hi, top in (
        (c - BLOCK_HALF, c - SLOT_HALF, BLOCK_TOP),
        (c + SLOT_HALF, c + BLOCK_HALF, BLOCK_TOP),
        (c - SLOT_HALF, c + SLOT_HALF, SLOT_FLOOR),
    ):
        ax.fill_between([lo, hi], BLOCK_BOTTOM, top, color="0.75", lw=0, zorder=2)
    if barrier:
        ax.plot([BARRIER_X, BARRIER_X], BARRIER_Y, ":", color="#008300", lw=2.5)
    ax.plot(GOAL_C, SLOT_FLOOR, "*", color="#008300", ms=12, zorder=6)
    if mu is not None:
        ax.plot(*(mu * SIZE).T, "-", color="#d03b3b", lw=0.8)
    ax.set(xlim=(0, SIZE), ylim=(0, SIZE), aspect="equal")


def draw_path(ax, log: np.ndarray, cfg: dict):
    path = log[:, :2] * SIZE
    line = LineCollection(np.stack([path[:-1], path[1:]], axis=1), cmap="viridis", lw=0.7)
    line.set_array(cfg["dt"] * np.arange(len(path) - 1))
    line.set_clim(0, cfg["T_max"])
    ax.add_collection(line)
    return line


def draw_trajectories(examples: dict[str, dict], cfg: dict, title: str) -> Figure:
    """One panel per condition: the kept episode's path coloured by time, its marks in red."""
    (_, mu, _), _, _ = setup(0, cfg)
    cols = min(len(examples), 4)
    rows_ = -(-len(examples) // cols)
    fig = Figure(figsize=(3.9 * cols, 3.7 * rows_), layout="constrained")
    fig.suptitle(title)
    axes = np.atleast_1d(fig.subplots(rows_, cols)).ravel()
    for ax, (name, e) in zip(axes, examples.items(), strict=False):
        log = np.array(e["log"])
        line = draw_path(ax, log, cfg)
        draw_world(ax, log[-1, 2], bool(log[-1, 3]), mu)
        for points, _, _, _, _ in e.get("marks", []):
            ax.plot(*(np.array(points) * SIZE).T, "-", color="#d03b3b", lw=2.5, alpha=0.6)
        outcome = f"done {e['time_to_success']:.0f} s" if e["success"] else "not done"
        ax.set_title(f"{name}: {outcome}, {e['entries']} entries, {e['n_marks']} marks", fontsize=9)
    for ax in axes[len(examples) :]:
        ax.set_visible(False)
    fig.colorbar(line, ax=axes, label="time (s)", shrink=0.7)
    return fig


def draw_phase(examples: dict[str, dict], cfg: dict, title: str) -> Figure:
    cols = min(len(examples), 3)
    rows_ = -(-len(examples) // cols)
    fig = Figure(figsize=(5 * cols, 3 * rows_), layout="constrained")
    fig.suptitle(title)
    axes = np.atleast_1d(fig.subplots(rows_, cols, sharey=True)).ravel()
    for ax, (name, e) in zip(axes, examples.items(), strict=False):
        trace, log = np.array(e["trace"]), np.array(e["log"])
        t = cfg["dt"] * np.arange(len(trace))
        for r in visits(log[:, 4] > 0):
            ax.axvspan(r.start * cfg["dt"], r.stop * cfg["dt"], color="0.88", lw=0)
        ax.plot(t, trace[:, 0], color=PALETTE[0], lw=1.0, label=r"$\hat\varphi$")
        ax.plot(t, log[:, 6], color=PALETTE[1], lw=1.0, label=r"$\tilde\varphi$")
        for t_peak, t_det, _ in e["firings"]:
            ax.axvline(t_det, color="#d03b3b", lw=0.8)
            ax.axvspan(t_peak, t_det, color="#d03b3b", alpha=0.12, lw=0)
        ax.set(title=name, xlabel="time (s)", ylim=(-0.02, 1.02), xlim=(0, max(t[-1], 1.0)))
    for ax in axes[len(examples) :]:
        ax.set_visible(False)
    axes[0].legend(frameon=False, loc="lower right")
    return fig


def draw_success(groups: dict[str, list[dict]], cfg: dict, title: str) -> Figure:
    fig = Figure(figsize=(7, 4.5), layout="constrained")
    ax = fig.subplots()
    t = np.linspace(0, cfg["T_max"], 601)
    for colour, (name, eps) in zip(PALETTE, groups.items(), strict=False):
        times = np.array([e["time_to_success"] for e in eps if e["success"]])
        ax.plot(t, [(times <= ti).sum() / len(eps) for ti in t], color=colour, lw=2, label=name)
    ax.set(xlabel="time (s)", ylabel="share of seeds done", ylim=(0, 1.02), title=title)
    ax.grid(True, color="0.9", lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    return fig
