"""The figures of results/README.md, from the milestones' tables and four re-run episodes.

- overview.png: success and time to success of every Scene B condition, right miss, 30 seeds;
- mechanism_trajectories.png, mechanism_phase.png: seed 3 (C3 fails there) under C2, C3, C5, C5g;
- c3_vs_c5.gif: the same seed, C3 beside C5, on one clock.
Run after m1..m4.
"""

import json
from pathlib import Path

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure

from experiments.sequential_ergodic.episode import CONFIG, Job, run_episode, setup
from experiments.sequential_ergodic.m3_marks import CONDITIONS as M3
from experiments.sequential_ergodic.report import PALETTE, draw_phase, draw_trajectories, draw_world
from simulation.peg_slide import SIZE

RESULTS = Path(__file__).parent / "results"
SEED = 3
SCENARIO = {"block_offset": -0.7}
# (label, milestone table, key in it, family)
OVERVIEW = [
    ("B0 full, static", "M1", "B0", "no marks"),
    ("B1 hard reset", "M1", "B1", "no marks"),
    ("C1 exp, static", "M1", "C1", "no marks"),
    ("C2 full, expanding", "M1", "C2", "no marks"),
    ("C3 exp", "M1", "C3", "no marks"),
    ("C3f freq", "M1", "C3f", "no marks"),
    ("C4 inside", "M3", "C4", "marks, exponential"),
    ("C5 separate", "M3", "C5", "marks, exponential"),
    ("C5g separate, global", "M3", "C5g", "marks, exponential"),
    ("C6a zeroed target", "M3", "C6a", "marks, exponential"),
    ("C6b negative weight", "M3", "C6b", "marks, exponential"),
    ("C4f inside", "M3", "C4f", "marks, frequency"),
    ("C5f separate", "M3", "C5f", "marks, frequency"),
    ("C5gf separate, global", "M3", "C5gf", "marks, frequency"),
    ("C6af zeroed target", "M3", "C6af", "marks, frequency"),
    ("C6bf negative weight", "M3", "C6bf", "marks, frequency"),
    ("C5s slider-capped phase", "M4", "C5s", "phase variants"),
    ("C5p tagged at the peak", "M4", "C5p", "phase variants"),
]
FAMILIES = {
    "no marks": PALETTE[0],
    "marks, exponential": PALETTE[1],
    "marks, frequency": PALETTE[2],
    "phase variants": PALETTE[4],
}


def _rows() -> dict[str, dict]:
    tables = {m: json.loads((RESULTS / m / "table.json").read_text()) for m in ("M1", "M3", "M4")}
    main = {
        "M1": tables["M1"]["main"],
        "M3": tables["M3"]["main"],
        "M4": tables["M4"]["right_miss"],
    }
    return {label: main[m][key] for label, m, key, _ in OVERVIEW}


def draw_overview() -> Figure:
    rows = _rows()
    fig = Figure(figsize=(11, 6.5), layout="constrained")
    fig.suptitle("Right miss on peg_slide, 30 seeds, lambda = 0.1: mean and 95 % bootstrap CI")
    axes = fig.subplots(1, 2, sharey=True)
    y = np.arange(len(OVERVIEW))[::-1]
    for yi, (label, _, _, family) in zip(y, OVERVIEW, strict=True):
        for ax, key in zip(axes, ("success_rate", "time_to_success"), strict=True):
            m, lo, hi = rows[label][key]
            if np.isnan(m):
                continue
            ax.errorbar(
                m, yi, xerr=[[m - lo], [hi - m]], fmt="o", ms=7, capsize=3, color=FAMILIES[family]
            )
    axes[0].set(
        yticks=y, yticklabels=[o[0] for o in OVERVIEW], xlim=(-0.03, 1.03), xlabel="success rate"
    )
    axes[1].set(xlabel="time to success, successful seeds only (s)", xlim=(0, None))
    c2 = rows["C2 full, expanding"]["time_to_success"][0]
    axes[1].axvline(c2, color="0.6", ls=":", lw=1)
    for ax in axes:
        ax.grid(True, axis="x", color="0.9", lw=0.6)
    handles = [axes[0].plot([], [], "o", color=c, label=f)[0] for f, c in FAMILIES.items()]
    axes[1].legend(handles=handles, frameon=False, loc="upper right")
    return fig


def animate(c3: dict, c5: dict, cfg: dict) -> FuncAnimation:
    (_, mu, _), _, _ = setup(SEED, cfg)
    fig = Figure(figsize=(9, 4.4), dpi=70, layout="constrained")
    axes = fig.subplots(1, 2)
    runs = (("C3, no marks", c3), ("C5, separate marks", c5))
    logs = [np.array(e["log"]) for _, e in runs]
    frames = range(0, max(len(log) for log in logs), 50)  # every 0.5 s

    def update(n: int) -> None:
        for ax, (name, e), log in zip(axes, runs, logs, strict=True):
            ax.clear()
            i = min(n, len(log) - 1)
            draw_world(ax, log[i, 2], bool(log[i, 3]), mu)
            ax.plot(*(log[: i + 1, :2] * SIZE).T, "-", color="0.45", lw=0.6)
            for points, _, _, _, t_fire in e["marks"]:
                if t_fire <= i * cfg["dt"]:
                    ax.plot(*(np.array(points) * SIZE).T, "-", color="#d03b3b", lw=3, alpha=0.6)
            ax.plot(*(log[i, :2] * SIZE), "o", color="k", ms=6, zorder=7)
            state = "done" if i == len(log) - 1 and e["success"] else f"t = {i * cfg['dt']:.0f} s"
            ax.set_title(f"{name}: {state}", fontsize=10)

    return FuncAnimation(fig, update, frames=frames)


def main() -> None:
    cfg = CONFIG
    names = ("C2", "C3", "C5", "C5g")
    eps = {
        n: run_episode(Job(n, M3[n], SCENARIO, None if n == "C2" else cfg["lam"], SEED, cfg, True))
        for n in names
    }
    draw_overview().savefig(RESULTS / "overview.png", dpi=150)
    title = f"Seed {SEED}, right miss (red: marks)"
    draw_trajectories(eps, cfg, title).savefig(RESULTS / "mechanism_trajectories.png", dpi=150)
    draw_phase(eps, cfg, f"Seed {SEED}: phase; grey = in the dead end, red = firings").savefig(
        RESULTS / "mechanism_phase.png", dpi=150
    )
    animate(eps["C3"], eps["C5"], cfg).save(RESULTS / "c3_vs_c5.gif", writer=PillowWriter(fps=12))
    print(
        {
            n: (e["success"], e["time_to_success"], e["entries"], e["n_marks"])
            for n, e in eps.items()
        }
    )


if __name__ == "__main__":
    main()
