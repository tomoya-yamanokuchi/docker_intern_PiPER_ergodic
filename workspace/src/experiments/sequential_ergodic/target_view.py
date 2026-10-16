"""The target density and the spatial statistic as the controller sees them, over one episode.

Case 2 (60 s of the first stage already in the statistic), forgetting + global marks (C5gf), seed 3.
Both are drawn from their Fourier coefficients, 1 + sum_k a_k F_k(x) with the K = 16 modes the law
uses, so the blur is the law's own. The agent moves to where the target (orange) exceeds the statistic
(blue). Writes results/target_view/: snapshots.png (five moments) and target_vs_statistic.gif.
"""

from pathlib import Path

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure

from experiments.sequential_ergodic.basis import make_basis, values
from experiments.sequential_ergodic.episode import CONFIG, Job, run_episode, setup
from experiments.sequential_ergodic.m3_marks import CONDITIONS
from experiments.sequential_ergodic.metrics import visits
from experiments.sequential_ergodic.report import draw_world
from simulation.peg_slide import SIZE

OUT = Path(__file__).parent / "results" / "target_view"
SEED, NAME = 3, "C5gf"
SCENARIO = {"block_offset": -0.7, "warmup_T": 60.0}
EVERY = 0.5  # s between snapshots
XS, YS = (
    np.linspace(0, SIZE, 121),
    np.linspace(0, 8, 97),
)  # world units; nothing happens above y = 8


def densities(snapshots: list, cfg: dict) -> tuple[np.ndarray, np.ndarray]:  # (T, Y, X) each
    """Target and statistic densities, 1 + sum_k a_k F_k(x), clipped at 0 where truncation dips below."""
    basis = make_basis(tuple(cfg["K"]))
    X, Y = np.meshgrid(XS, YS)
    V = values(basis, np.stack([X, Y], axis=-1) / SIZE)  # (Y, X, N)
    p = np.array([s[1] for s in snapshots]) @ V.reshape(-1, V.shape[-1]).T
    c = np.array([s[2] for s in snapshots]) @ V.reshape(-1, V.shape[-1]).T
    shape = (len(snapshots), *X.shape)
    return np.clip(1 + p, 0, None).reshape(shape), np.clip(1 + c, 0, None).reshape(shape)


def moments(e: dict, cfg: dict) -> list[tuple[float, str]]:
    """Five times that tell the story: start, deepest in the dead end, backed out, retrying, done."""
    log, dt = np.array(e["log"]), cfg["dt"]
    phi = log[:, 6]
    first = visits(log[:, 4] > 0)[0]
    deep = first.start + int(phi[first].argmax())
    back = first.stop + int(phi[first.stop :].argmin())
    end = len(log) - 1
    return [
        (0.0, "start\nwarm-up still remembered"),
        (deep * dt, "in the dead end\ntarget grew along the push line"),
        (back * dt, "backed out\ntarget shrank to the start"),
        ((back + end) / 2 * dt, "retry\nmark keeps the dead end covered"),
        (end * dt, "done\nblock at the goal"),
    ]


def _panel(ax, image: tuple, log: np.ndarray, i: int, mu: np.ndarray, marks=None) -> None:
    """image = (field (Y, X), vmax, colormap); marks given: also the path so far and the marks."""
    field, vmax, cmap = image
    ax.imshow(
        field, origin="lower", extent=(0, SIZE, 0, 8), cmap=cmap, vmin=0, vmax=vmax, aspect="equal"
    )
    draw_world(ax, log[i, 2], bool(log[i, 3]))
    ax.plot(*(mu * SIZE).T, "--", color="#5a636c", lw=0.9)
    if marks is not None:
        ax.plot(*(log[: i + 1, :2] * SIZE).T, "-", color="#1b2026", lw=0.6, alpha=0.6)
        for points, _, _, _, t_fire in marks:
            if t_fire <= i * CONFIG["dt"]:
                ax.plot(*(np.array(points) * SIZE).T, "-", color="#d03b3b", lw=3, alpha=0.7)
    ax.plot(*(log[i, :2] * SIZE), "o", color="#1b2026", mec="#ffffff", mew=1.5, ms=8, zorder=8)
    ax.set(xlim=(0, SIZE), ylim=(0, 8), xticks=[], yticks=[])


def draw_snapshots(e: dict, p: np.ndarray, c: np.ndarray, cfg: dict) -> Figure:
    (_, mu, _), _, _ = setup(SEED, cfg)
    log = np.array(e["log"])
    vp, vc = np.percentile(p, 99.5), np.percentile(c, 99.5)
    shots = moments(e, cfg)
    fig = Figure(figsize=(16, 6.2), dpi=120, layout="constrained", facecolor="#ffffff")
    axes = fig.subplots(2, len(shots))
    for j, (t, label) in enumerate(shots):
        k = min(round(t / EVERY), len(p) - 1)
        i = min(round(t / cfg["dt"]), len(log) - 1)
        _panel(axes[0, j], (p[k], vp, "Oranges"), log, i, mu)
        _panel(axes[1, j], (c[k], vc, "Blues"), log, i, mu, e["marks"])
        head, rest = label.split("\n")
        axes[0, j].set_title(f"t = {t:.0f} s · {head}\n{rest}", fontsize=11, loc="left")
    axes[0, 0].set_ylabel("target density p", fontsize=11)
    axes[1, 0].set_ylabel("spatial statistic ĉ", fontsize=11)
    return fig


def animate(e: dict, p: np.ndarray, c: np.ndarray, cfg: dict) -> FuncAnimation:
    (_, mu, _), _, _ = setup(SEED, cfg)
    log = np.array(e["log"])
    vp, vc = np.percentile(p, 99.5), np.percentile(c, 99.5)
    fig = Figure(figsize=(12, 6.2), dpi=72, layout="constrained", facecolor="#ffffff")
    grid = fig.add_gridspec(2, 2, height_ratios=[4, 1])
    target_ax, stat_ax, phase_ax = (
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :]),
    )
    t_log = cfg["dt"] * np.arange(len(log))

    def update(k: int) -> None:
        i = min(round(k * EVERY / cfg["dt"]), len(log) - 1)
        for ax in (target_ax, stat_ax, phase_ax):
            ax.clear()
        _panel(target_ax, (p[k], vp, "Oranges"), log, i, mu)
        _panel(stat_ax, (c[k], vc, "Blues"), log, i, mu, e["marks"])
        target_ax.set_title(f"target density p  (t = {k * EVERY:.1f} s)", fontsize=13, loc="left")
        stat_ax.set_title("spatial statistic ĉ, with marks (red)", fontsize=13, loc="left")
        for r in visits(log[:, 4] > 0):
            phase_ax.axvspan(r.start * cfg["dt"], r.stop * cfg["dt"], color="#f6d9d6", lw=0)
        phase_ax.plot(t_log, log[:, 6], color="#1b2026", lw=1.2)
        phase_ax.axvline(k * EVERY, color="#c2551f", lw=2)
        phase_ax.set(
            xlim=(0, t_log[-1]),
            ylim=(-0.05, 1.05),
            yticks=[0, 1],
            xlabel="time (s)",
            ylabel="phase",
        )

    return FuncAnimation(fig, update, frames=range(len(p)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG
    e = run_episode(Job(NAME, CONDITIONS[NAME], SCENARIO, cfg["lam"], SEED, cfg, True, EVERY))
    p, c = densities(e["snapshots"], cfg)
    draw_snapshots(e, p, c, cfg).savefig(OUT / "snapshots.png", facecolor="#ffffff")
    animate(e, p, c, cfg).save(OUT / "target_vs_statistic.gif", writer=PillowWriter(fps=6))
    print(
        f"{NAME} seed {SEED}: done={e['success']} in {e['time_to_success']} s, {len(p)} snapshots"
    )
    print([(round(t, 1), label) for t, label in moments(e, cfg)])


if __name__ == "__main__":
    main()
