"""Live view of the ten-trial experiment: full memory beside forgetting + marks, on one clock.

    python -m experiments.sequential_ergodic.live_view                 # a window to screen-record
    python -m experiments.sequential_ergodic.live_view --speed 8      # faster than the default 5x
    python -m experiments.sequential_ergodic.live_view --save run.gif  # the same, written to a GIF

Seed 7 of showoff.py, block 0.7 short. Both runs are simulated first (a few seconds), then played:
the orange background is the target density the law is asked to cover, drawn from its K = 16
Fourier modes; red lines are the marks so far; the row below each panel scores the trials.
"""

import argparse
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from experiments.sequential_ergodic.basis import make_basis, values
from experiments.sequential_ergodic.episode import CONFIG, Job, setup
from experiments.sequential_ergodic.showoff import CONDITIONS, N_TRIALS, OFFSET, SEED, run_trials
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

NAMES = ("full memory", "forgetting + marks")
COLOURS = {"full memory": "#2a78d6", "forgetting + marks": "#0e6a70"}
FRAME = 0.2  # s of simulated time between frames
XS, YS = np.linspace(0, SIZE, 81), np.linspace(0, 8, 65)


@dataclass
class Run:
    """One method's ten trials as frames on one clock."""

    name: str
    t: np.ndarray  # (F,) s since the first trial began
    trial: np.ndarray  # (F,) trial index of each frame
    xy: np.ndarray  # (F, 2) world
    block: np.ndarray  # (F,) block centre
    barrier: np.ndarray  # (F,) bool
    p: np.ndarray  # (F, N) target coefficients
    results: list[tuple[bool, float, float]]  # per trial: success, its time, the clock at its end
    marks: list[tuple[np.ndarray, float]]  # (points (P, 2) world, firing time on the clock)


def simulate(name: str, cfg: dict) -> Run:
    cond = CONDITIONS[name]
    lam = None if cond["statistic"] == "full" else cfg["lam"]
    run = run_trials(Job(name, cond, {"block_offset": OFFSET}, lam, SEED, cfg, True, FRAME))
    frames = [(k, *f) for k, tr in enumerate(run["trials"]) for f in tr["frames"]]
    k, t, x, y, c, barrier = (np.array([f[i] for f in frames]) for i in range(6))
    results, clock = [], 0.0
    for tr in run["trials"]:
        clock += tr["time"]
        results.append((tr["success"], tr["time"], clock))
    marks = []
    if cond.get("marks"):  # marks only act where the method uses them
        marks = [(np.array(pts) * SIZE, t_fire) for pts, t_fire in run["trials"][-1]["marks"]]
    return Run(
        name,
        t,
        k.astype(int),
        np.column_stack([x, y]) * SIZE,
        c,
        barrier.astype(bool),
        np.array([f[6] for f in frames]),
        results,
        marks,
    )


class Panel:
    """The artists of one method's panel, updated in place every frame."""

    def __init__(self, ax, score_ax, run: Run, mu: np.ndarray, vmax: float):
        self.run, self.ax = run, ax
        self.image = ax.imshow(
            np.zeros((len(YS), len(XS))),
            origin="lower",
            extent=(0, SIZE, 0, 8),
            cmap="Oranges",
            vmin=0,
            vmax=vmax,
            aspect="equal",
            zorder=0,
        )
        self.parts = [Rectangle((0, BLOCK_BOTTOM), 1, 1, color="0.62", zorder=2) for _ in range(3)]
        for part in self.parts:
            ax.add_patch(part)
        (self.barrier,) = ax.plot(
            [BARRIER_X, BARRIER_X], BARRIER_Y, ":", color="#008300", lw=3, zorder=3
        )
        ax.plot(GOAL_C, SLOT_FLOOR, "*", color="#008300", ms=18, zorder=6)
        ax.plot(*(mu * SIZE).T, "--", color="#5a636c", lw=1.0, zorder=1)
        self.marks = LineCollection([], colors="#c0261f", lw=4, alpha=0.6, zorder=4)
        ax.add_collection(self.marks)
        (self.path,) = ax.plot([], [], "-", color="#1b2026", lw=1.2, zorder=5)
        (self.peg,) = ax.plot([], [], "o", color="#1b2026", mec="#ffffff", mew=2, ms=11, zorder=7)
        ax.set(xlim=(0, SIZE), ylim=(0, 8), xticks=[], yticks=[])
        for side in ax.spines.values():
            side.set_color(COLOURS[run.name])
            side.set_linewidth(3)
        score_ax.set_axis_off()
        score_ax.set(xlim=(0, N_TRIALS), ylim=(0, 1))
        self.scores = [
            score_ax.text(
                i + 0.5, 0.3, "", ha="center", va="center", fontsize=12, fontweight="bold"
            )
            for i in range(N_TRIALS)
        ]
        for i in range(N_TRIALS):
            score_ax.text(
                i + 0.5,
                0.85,
                f"trial {i + 1}",
                ha="center",
                va="center",
                fontsize=10,
                color="#8a949d",
            )
        self.title = ax.set_title(
            "", fontsize=16, loc="left", color=COLOURS[run.name], fontweight="bold"
        )

    def update(self, t: float, basis_grid: np.ndarray) -> None:
        run = self.run
        i = min(int(np.searchsorted(run.t, t, side="right")) - 1, len(run.t) - 1)
        k, c = run.trial[i], run.block[i]
        self.image.set_data(np.clip(1 + basis_grid @ run.p[i], 0, None).reshape(len(YS), len(XS)))
        for part, (lo, hi, top) in zip(
            self.parts,
            (
                (c - BLOCK_HALF, c - SLOT_HALF, BLOCK_TOP),
                (c + SLOT_HALF, c + BLOCK_HALF, BLOCK_TOP),
                (c - SLOT_HALF, c + SLOT_HALF, SLOT_FLOOR),
            ),
            strict=True,
        ):
            part.set_bounds(lo, BLOCK_BOTTOM, hi - lo, top - BLOCK_BOTTOM)
        self.barrier.set_visible(bool(run.barrier[i]))
        in_trial = (run.trial == k) & (run.t <= t)
        self.path.set_data(*run.xy[in_trial].T)
        self.peg.set_data([run.xy[i, 0]], [run.xy[i, 1]])
        self.marks.set_segments([pts for pts, t_fire in run.marks if t_fire <= t])
        finished = t >= run.results[-1][2]
        state = (
            f"{sum(r[0] for r in run.results)}/10 done, {run.results[-1][2]:.0f} s for all ten"
            if finished
            else f"trial {k + 1} of {N_TRIALS},  {t - (run.results[k][2] - run.results[k][1]):4.0f} s"
        )
        self.title.set_text(f"{run.name}:  {state}")
        for j, (success, time, end) in enumerate(run.results):
            text, colour = ("", "#1b2026")
            if end <= t:
                text, colour = (f"{time:.0f} s", "#1f7a3f") if success else ("FAIL", "#b0322a")
            elif j == k and not finished:
                text, colour = ("...", "#59636e")
            self.scores[j].set_text(text)
            self.scores[j].set_color(colour)


def build(cfg: dict, speed: float):
    runs = [simulate(name, cfg) for name in NAMES]
    (_, mu, _), _, _ = setup(SEED, cfg)
    X, Y = np.meshgrid(XS, YS)
    grid = values(make_basis(tuple(cfg["K"])), np.stack([X, Y], axis=-1).reshape(-1, 2) / SIZE)
    vmax = max(np.percentile(1 + grid @ r.p[:: max(1, len(r.p) // 200)].T, 99.5) for r in runs)
    fig = plt.figure(figsize=(16, 9), facecolor="#f4f6f7")
    fig.suptitle(
        "Ten trials, one memory: the demonstration is wrong about where the block is",
        fontsize=20,
        x=0.03,
        ha="left",
        y=0.97,
    )
    grid_spec = fig.add_gridspec(
        2,
        2,
        height_ratios=[8, 1],
        left=0.03,
        right=0.97,
        top=0.88,
        bottom=0.08,
        wspace=0.06,
        hspace=0.05,
    )
    panels = [
        Panel(fig.add_subplot(grid_spec[0, j]), fig.add_subplot(grid_spec[1, j]), r, mu, vmax)
        for j, r in enumerate(runs)
    ]
    clock = fig.text(0.03, 0.025, "", fontsize=13, color="#59636e")
    legend = "orange: target density  ·  red: marks  ·  black: this trial's path  ·  dotted green: barrier"
    fig.text(0.97, 0.025, legend, fontsize=13, color="#59636e", ha="right")
    end = max(r.results[-1][2] for r in runs) + 3.0

    def update(f: int) -> None:
        t = f * FRAME
        for panel in panels:
            panel.update(t, grid)
        clock.set_text(f"simulated time {t:5.0f} s   ({speed:g}x real time)")

    frames = range(int(end / FRAME) + 1)
    return fig, FuncAnimation(
        fig, update, frames=frames, interval=1000 * FRAME / speed, repeat=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--speed", type=float, default=5.0, help="times real time (default 5)")
    parser.add_argument("--save", help="write a GIF here instead of opening a window")
    args = parser.parse_args()
    if args.save:
        plt.switch_backend("Agg")
    fig, anim = build(CONFIG, args.speed)
    if args.save:
        fig.set_dpi(60)
        anim.save(args.save, writer=PillowWriter(fps=round(args.speed / FRAME)))
    else:
        plt.show()


if __name__ == "__main__":
    main()
