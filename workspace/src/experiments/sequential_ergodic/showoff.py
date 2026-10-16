"""Learning from failure across trials: the peg-slide task with a misplaced block, ten trials in a row.

The demonstration is wrong about where the block is (it starts block_offset from where it was taught),
so following it fails. Between trials the peg goes back to the start and the block back home, as
on the arm, but the controller keeps its memory: the spatial statistic, the marks, the clock. The
measure is time to success per trial (the E2T2 trials of CLAUDE.md section 3).

The 2x2 ablation: {full memory, forgetting} x {no marks, global marks}.
"""

import json
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from experiments.sequential_ergodic import scene_b
from experiments.sequential_ergodic.basis import make_basis
from experiments.sequential_ergodic.controller import Plant, trajectory
from experiments.sequential_ergodic.detector import NegativeProgressDetector
from experiments.sequential_ergodic.episode import CONFIG, Job, _phase_target, setup, statistic_for
from experiments.sequential_ergodic.marks import MarkingTarget, MarkParams
from experiments.sequential_ergodic.metrics import bootstrap_ci, visits
from experiments.sequential_ergodic.report import draw_world, run_all
from simulation.peg_slide import C0, SIZE, Slide

N_TRIALS = 10
_MARKS = {"expanding": True, "marks": "separate_global"}
CONDITIONS = {
    "full memory": {"statistic": "full", "expanding": True},
    "full memory + marks": {"statistic": "full"} | _MARKS,
    "forgetting": {"statistic": "exponential", "expanding": True},
    "forgetting + marks": {"statistic": "exponential"} | _MARKS,
    "frequency forgetting + marks": {"statistic": "frequency"} | _MARKS,
}


def _one_trial(
    target: MarkingTarget, x0: np.ndarray, home: float, cfg: dict, every: int, t_start: float
) -> tuple[np.ndarray, list, bool]:
    """One trial from x0 with the block at home; the target's statistic and marks carry over.

    every > 0 also records a frame every that many steps: (time since the first trial, x, y,
    block centre, barrier, target coefficients p_k).
    """
    world = Slide(home=home)
    target.inner.estimator.reset(x0)
    target.detector = NegativeProgressDetector(**cfg["detector"])
    plant = Plant(lambda z, d: scene_b.move(world, z, d), cfg["dt"], cfg["u_max"])
    log, frames = [], []  # log: x, y, block centre, in the dead end, phi_tilde
    for x, _, p in trajectory(target.basis, target.statistic, target, x0, plant):
        log.append((*x, world.c, scene_b.in_dead_end(world, x), target.inner.trace[-1][1]))
        if every and (len(log) - 1) % every == 0:
            t = t_start + (len(log) - 1) * cfg["dt"]
            frames.append((t, *x, world.c, world.barrier, p.astype(np.float32)))
        if scene_b.done(world, x) or len(log) * cfg["dt"] >= cfg["T_max"]:
            break
    return np.array(log, dtype=float), frames, scene_b.done(world, x)


def run_trials(job: Job) -> dict:
    """N_TRIALS trials on one seed; the controller's memory carries over, the world is reset."""
    cfg, cond = job.cfg, job.condition
    basis = make_basis(tuple(cfg["K"]))
    master, x0, _ = setup(job.seed, cfg)
    home = C0 + job.scenario["block_offset"]
    model, phase_target = _phase_target(job, basis, master, x0, Slide(home=home))
    statistic = statistic_for(cond["statistic"], job.lam, basis, model, cfg)
    target = MarkingTarget(
        phase_target,
        statistic,
        basis,
        NegativeProgressDetector(**cfg["detector"]),
        cond.get("marks"),
        MarkParams(**cfg["marks"]),
    )
    every = round(job.snapshot_every / cfg["dt"])
    trials, t = [], 0.0
    for _ in range(N_TRIALS):
        n_marks = len(target.marks)
        log, frames, success = _one_trial(target, x0, home, cfg, every, t)
        t += len(log) * cfg["dt"]
        trial = {
            "success": success,
            "time": len(log) * cfg["dt"],
            "entries": len(visits(log[:, 3] > 0)),
            "new_marks": len(target.marks) - n_marks,
        }
        if job.keep_log:
            trial["log"] = log[::5].tolist()  # every 5th step is plenty to draw
            # every mark so far, the trial's own last; firing times run on one clock over the trials
            trial["marks"] = [(m.points[::10].tolist(), m.t_fire) for m in target.marks]
        if frames:
            trial["frames"] = frames
        trials.append(trial)
    return {"name": job.name, "seed": job.seed, "trials": trials}


def jobs(cfg: dict, seeds: range, offset: float, keep_seed: int | None = None) -> list[Job]:
    return [
        Job(
            name,
            cond,
            {"block_offset": offset},
            None if cond["statistic"] == "full" else cfg["lam"],
            s,
            cfg,
            s == keep_seed,
        )
        for name, cond in CONDITIONS.items()
        for s in seeds
    ]


OUT = Path(__file__).parent / "results" / "showoff"
OFFSET = -0.7
SEED = 7  # the first seed whose full-memory failure count (2 of 10) is the median over the 30 seeds
STRIP_TRIALS = (1, 4, 5, 7, 10)
STRIP = ("full memory", "forgetting + marks")
STYLE = {  # colour, line width, dashes
    "full memory": ("#2a78d6", 2.0, ""),
    "full memory + marks": ("#7b5cd6", 2.0, ""),
    "forgetting": ("#eb6834", 2.0, ""),
    "forgetting + marks": ("#0e6a70", 4.0, ""),
    "frequency forgetting + marks": ("#0e6a70", 2.5, "--"),
}


def _strip_panel(ax, trial: dict, k: int, mu: np.ndarray, cfg: dict, marked: bool) -> None:
    log = np.array(trial["log"])
    path = log[:, :2] * SIZE
    draw_world(ax, log[-1, 2], False)
    ax.plot(*(mu * SIZE).T, "--", color="#8a949d", lw=0.8)
    line = LineCollection(np.stack([path[:-1], path[1:]], axis=1), cmap="viridis", lw=1.0)
    line.set_array(5 * cfg["dt"] * np.arange(len(path) - 1))
    line.set_clim(0, cfg["T_max"])
    ax.add_collection(line)
    if marked:  # every mark so far; the trial's own ones darker
        first_new = len(trial["marks"]) - trial["new_marks"]
        for i, (points, _) in enumerate(trial["marks"]):
            new = i >= first_new
            colour, alpha = ("#c0261f", 0.75) if new else ("#e8908a", 0.55)
            ax.plot(*(np.array(points) * SIZE).T, "-", color=colour, lw=3, alpha=alpha)
    ax.set(xlim=(0, SIZE), ylim=(0, 8), xticks=[], yticks=[])
    done = f"done in {trial['time']:.0f} s" if trial["success"] else "FAILED (120 s)"
    ax.set_title(
        f"trial {k}: {done}",
        fontsize=10,
        loc="left",
        color="#1b2026" if trial["success"] else "#b0322a",
    )


def _curves(ax_success, ax_time, results: list[dict]) -> None:
    rng = np.random.default_rng(0)
    trials = np.arange(1, N_TRIALS + 1)
    for name, (colour, width, dashes) in STYLE.items():
        runs = [r for r in results if r["name"] == name]
        success = np.array([[t["success"] for t in r["trials"]] for r in runs], dtype=float)
        times = np.array([[t["time"] for t in r["trials"]] for r in runs])
        for ax, values in ((ax_success, success), (ax_time, times)):
            m, lo, hi = np.array([bootstrap_ci(values[:, k], rng) for k in range(N_TRIALS)]).T
            ax.fill_between(trials, lo, hi, color=colour, alpha=0.12, lw=0)
            ax.plot(
                trials,
                m,
                color=colour,
                lw=width,
                ls=dashes or "-",
                marker="o",
                ms=4,
                label=f"{name} ({int(success.sum())}/{success.size})",
            )
    ax_success.set(ylabel="success rate", ylim=(0.0, 1.03), xticks=trials)
    ax_time.set(
        ylabel="time per trial (s)\nfailure counts 120 s",
        xlabel="trial",
        xticks=trials,
        ylim=(0, None),
    )
    ax_success.legend(frameon=False, fontsize=9, loc="lower left")
    for ax in (ax_success, ax_time):
        ax.grid(True, color="0.9", lw=0.6)


def draw(results: list[dict], cfg: dict) -> Figure:
    (_, mu, _), _, _ = setup(SEED, cfg)
    fig = Figure(figsize=(16, 6.4), dpi=120, layout="constrained", facecolor="#ffffff")
    outer = fig.add_gridspec(1, 2, width_ratios=[2.15, 1])
    strip = outer[0].subgridspec(2, len(STRIP_TRIALS))
    for i, name in enumerate(STRIP):
        run = next(r for r in results if r["name"] == name and r["seed"] == SEED)
        for j, k in enumerate(STRIP_TRIALS):
            ax = fig.add_subplot(strip[i, j])
            marked = CONDITIONS[name].get("marks") is not None
            _strip_panel(ax, run["trials"][k - 1], k, mu, cfg, marked)
            if j == 0:
                ax.set_ylabel(name, fontsize=12, color=STYLE[name][0], fontweight="bold")
    fig.text(
        0.01,
        0.005,
        f"Seed {SEED}, block 0.7 short. Path colour = time in the trial. Red: all marks so far, darker if placed in that trial. Full memory uses no marks.",
        fontsize=10,
        color="#59636e",
    )
    curves = outer[1].subgridspec(2, 1)
    _curves(fig.add_subplot(curves[0]), fig.add_subplot(curves[1]), results)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG
    results = run_all(jobs(cfg, range(cfg["seeds"]), OFFSET, keep_seed=SEED), run_trials)
    light = [
        {
            **r,
            "trials": [
                {k: v for k, v in t.items() if k not in ("log", "marks", "frames")}
                for t in r["trials"]
            ],
        }
        for r in results
    ]
    (OUT / "results.json").write_text(
        json.dumps({"offset": OFFSET, "seed_shown": SEED, "runs": light})
    )
    draw(results, cfg).savefig(OUT / "showoff.png", facecolor="#ffffff")
    for name in CONDITIONS:
        success = np.array(
            [[t["success"] for t in r["trials"]] for r in light if r["name"] == name]
        )
        print(
            f"{name:30s} {int(success.sum())}/{success.size}  per trial {success.mean(axis=0).round(2)}"
        )


if __name__ == "__main__":
    main()
