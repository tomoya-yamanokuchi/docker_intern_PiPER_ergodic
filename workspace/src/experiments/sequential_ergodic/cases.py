"""Five cases on peg_slide, each as one figure: the same seed under three conditions, with 30-seed totals.

Every case uses seed 3 (the first seed where C3 fails in M3) so the panels are not picked per case;
each panel's title also gives that condition's result over all 30 seeds. Writes results/cases/:
<case>.png, case_saturated.gif and summary.json. It runs its own episodes.
"""

import json
from pathlib import Path

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from experiments.sequential_ergodic.episode import CONFIG, Job, setup
from experiments.sequential_ergodic.m3_marks import CONDITIONS as M3
from experiments.sequential_ergodic.m4_peg_slide import CONDITIONS as M4
from experiments.sequential_ergodic.metrics import visits
from experiments.sequential_ergodic.report import draw_world, run_all
from simulation.peg_slide import BLOCK_BOTTOM, BLOCK_HALF, BLOCK_TOP, SIZE

OUT = Path(__file__).parent / "results" / "cases"
SEED = 3
CONDITIONS = M3 | M4
LABELS = {
    "C2": "full memory",
    "C3": "forgetting, no marks",
    "C5gf": "forgetting + global marks",
    "C5": "phase-conditioned marks",
    "C5g": "global marks",
    "C5s": "phase-conditioned + slider state",
}
_MAIN = ("C2", "C3", "C5gf")
CASES = {
    "fresh": ("Dead end, fresh start", {"block_offset": -0.7}, _MAIN),
    "saturated": (
        "Dead end after 60 s in the first stage",
        {"block_offset": -0.7, "warmup_T": 60.0},
        _MAIN,
    ),
    "control": ("Control: no dead end", {"block_offset": 0.7}, _MAIN),
    "incomplete_push": (
        "Block pushed 40 % and left",
        {"block_offset": 0.0, "prefix": "incomplete_push", "push_fraction": 0.4},
        _MAIN,
    ),
    "mark_kinds": ("Kinds of marks, dead end", {"block_offset": -0.7}, ("C5", "C5g", "C5s")),
}
INK, MUTED, MARK, MASTER = "#1b2026", "#59636e", "#d03b3b", "#8a949d"


def jobs(cfg: dict) -> list[Job]:
    out = []
    for case, (_, scenario, names) in CASES.items():
        for n in names:
            lam = None if n == "C2" else cfg["lam"]
            out += [
                Job(f"{case}/{n}", CONDITIONS[n], scenario, lam, s, cfg, s == SEED)
                for s in range(cfg["seeds"])
            ]
    return out


def totals(eps: list[dict]) -> dict:
    times = [e["time_to_success"] for e in eps if e["success"]]
    return {
        "done": len(times),
        "n": len(eps),
        "median_time": float(np.median(times)) if times else None,
    }


def _draw_panel(ax_world, ax_phase, e: dict, total: dict, name: str, cfg: dict) -> LineCollection:
    (_, mu, _), _, _ = setup(SEED, cfg)
    log = np.array(e["log"])
    path = log[:, :2] * SIZE
    home = log[0, 2]
    draw_world(ax_world, log[-1, 2], bool(log[-1, 3]))
    if abs(log[-1, 2] - home) > 1e-6:  # where the block was when the controller started
        ax_world.add_patch(
            Rectangle(
                (home - BLOCK_HALF, BLOCK_BOTTOM),
                2 * BLOCK_HALF,
                BLOCK_TOP - BLOCK_BOTTOM,
                fill=False,
                ls="--",
                lw=1.2,
                ec=MUTED,
                zorder=2,
            )
        )
    ax_world.plot(*(mu * SIZE).T, "--", color=MASTER, lw=1.0, zorder=1)
    line = LineCollection(np.stack([path[:-1], path[1:]], axis=1), cmap="viridis", lw=1.1, zorder=4)
    line.set_array(cfg["dt"] * np.arange(len(path) - 1))
    line.set_clim(0, cfg["T_max"])
    ax_world.add_collection(line)
    for points, _, _, _, _ in e["marks"]:
        ax_world.plot(*(np.array(points) * SIZE).T, "-", color=MARK, lw=4, alpha=0.45, zorder=3)
    ax_world.plot(*path[0], "o", color=INK, ms=7, zorder=6)
    ax_world.set(xlim=(0, SIZE), ylim=(0, 8), xticks=[], yticks=[])
    outcome = (
        f"done in {e['time_to_success']:.0f} s"
        if e["success"]
        else f"not done in {cfg['T_max']:.0f} s"
    )
    ax_world.set_title(
        f"{name}: {LABELS[name]}\nthis seed: {outcome}  ·  all seeds: {total['done']}/{total['n']}",
        fontsize=11,
        color=INK,
        loc="left",
    )
    t = cfg["dt"] * np.arange(len(log))
    for r in visits(log[:, 4] > 0):
        ax_phase.axvspan(r.start * cfg["dt"], r.stop * cfg["dt"], color="#f6d9d6", lw=0)
    ax_phase.plot(t, log[:, 6], color=INK, lw=1.0)
    for _, t_det, _ in e["firings"]:
        ax_phase.axvline(t_det, color=MARK, lw=0.8)
    ax_phase.set(xlim=(0, cfg["T_max"]), ylim=(-0.05, 1.05), yticks=[0, 1], xlabel="time (s)")
    ax_phase.set_ylabel("phase", fontsize=9)
    ax_phase.tick_params(labelsize=8)
    return line


def draw_case(case: str, examples: dict[str, dict], tots: dict[str, dict], cfg: dict) -> Figure:
    names = CASES[case][2]  # the case's title goes on its slide
    fig = Figure(figsize=(16, 6.4), dpi=120, layout="constrained", facecolor="#ffffff")
    grid = fig.add_gridspec(2, len(names), height_ratios=[4.3, 1])
    worlds = []
    for j, n in enumerate(names):
        worlds.append(fig.add_subplot(grid[0, j]))
        line = _draw_panel(worlds[-1], fig.add_subplot(grid[1, j]), examples[n], tots[n], n, cfg)
    bar = fig.colorbar(line, ax=worlds, shrink=0.7, pad=0.01, aspect=30)
    bar.set_label("time on the path (s)", fontsize=10)
    fig.get_layout_engine().set(w_pad=0.15)
    return fig


def animate(examples: dict[str, dict], names: tuple, cfg: dict) -> FuncAnimation:
    (_, mu, _), _, _ = setup(SEED, cfg)
    fig = Figure(figsize=(13.5, 4.4), dpi=72, layout="constrained", facecolor="#ffffff")
    axes = fig.subplots(1, len(names))
    logs = [np.array(examples[n]["log"]) for n in names]
    frames = range(0, max(len(log) for log in logs), 50)

    def update(i: int) -> None:
        for ax, n, log in zip(axes, names, logs, strict=True):
            ax.clear()
            k = min(i, len(log) - 1)
            draw_world(ax, log[k, 2], bool(log[k, 3]))
            ax.plot(*(mu * SIZE).T, "--", color=MASTER, lw=1.0)
            ax.plot(*(log[: k + 1, :2] * SIZE).T, "-", color="#4a6fa5", lw=0.8)
            for points, _, _, _, t_fire in examples[n]["marks"]:
                if t_fire <= k * cfg["dt"]:
                    ax.plot(*(np.array(points) * SIZE).T, "-", color=MARK, lw=4, alpha=0.45)
            ax.plot(*(log[k, :2] * SIZE), "o", color=INK, ms=7)
            ax.set(xlim=(0, SIZE), ylim=(0, 8), xticks=[], yticks=[])
            done = k == len(log) - 1 and examples[n]["success"]
            ax.set_title(
                f"{LABELS[n]}: " + ("done" if done else f"t = {k * cfg['dt']:.0f} s"),
                fontsize=12,
                loc="left",
            )

    return FuncAnimation(fig, update, frames=frames)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG
    episodes = run_all(jobs(cfg))
    summary = {}
    for case, (_, _, names) in CASES.items():
        groups = {n: [e for e in episodes if e["name"] == f"{case}/{n}"] for n in names}
        tots = {n: totals(eps) for n, eps in groups.items()}
        examples = {n: next(e for e in eps if e["seed"] == SEED) for n, eps in groups.items()}
        draw_case(case, examples, tots, cfg).savefig(OUT / f"{case}.png", facecolor="#ffffff")
        if case == "saturated":
            animate(examples, names, cfg).save(
                OUT / "case_saturated.gif", writer=PillowWriter(fps=12)
            )
        summary[case] = {
            n: tots[n]
            | {
                "seed": SEED,
                "this_seed_done": examples[n]["success"],
                "this_seed_time": examples[n]["time_to_success"],
                "this_seed_entries": examples[n]["entries"],
                "this_seed_marks": examples[n]["n_marks"],
            }
            for n in names
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
