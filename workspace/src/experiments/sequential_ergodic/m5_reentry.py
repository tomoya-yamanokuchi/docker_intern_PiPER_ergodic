"""M5: does forgetting give a reason to go through the sequence again, where full memory does not?

The right miss of M1-M4 (block 0.7 short), with T_pre seconds of the first stage already in the
statistic: a scripted peg wanders the master's way down, above the block, and its states enter the
statistic as visits (episode.warm_up). Then the controller starts at the master's start as before.

Holding the controller itself in the first stage was tried first and dropped: a small held target is
covered quickly and the law then leaves it, so 30-70 % of warm-ups reached the dead end, and the held
phase fired the stall detector every ~6 s.

The claim predicts that full memory (C2), saturated with the first stage, stops going back to it after
the dead end as T_pre grows, while forgetting (C3, C5g, C5gf) forgets the warm-up and keeps re-entering.
Measured: success within 120 s and its time; the dead end; whether, after leaving the dead end,
phi_tilde goes back to the start of the sequence (<= 0.05) and how soon (re-entry delay).

Writes results/M5/: config.json, episodes.json, table.json and the figures.
"""

import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from experiments.sequential_ergodic.episode import CONFIG, Job, run_episode
from experiments.sequential_ergodic.m3_marks import CONDITIONS as M3
from experiments.sequential_ergodic.metrics import bootstrap_ci, visits
from experiments.sequential_ergodic.report import PALETTE, draw_phase, draw_trajectories, run_all

OUT = Path(__file__).parent / "results" / "M5"
T_PRE = [0.0, 10.0, 30.0, 60.0]
START = 0.05  # phi_tilde at or below this is back at the start of the sequence
NAMES = ("C2", "C3", "C5g", "C5gf")


def reentry_outcome(job: Job) -> dict:
    """The episode's outcome and its re-entry; the log only for seed 0."""
    e = run_episode(Job(job.name, job.condition, job.scenario, job.lam, job.seed, job.cfg, True))
    dt, T = job.cfg["dt"], job.scenario["warmup_T"]
    post = np.array(e["log"])
    runs = visits(post[:, 4] > 0)
    first_exit = runs[0].stop if runs and runs[0].stop < len(post) else None
    back = None
    if first_exit is not None:
        low = np.nonzero(post[first_exit:, 6] <= START)[0]
        back = float(low[0] * dt) if len(low) else None
    t_success = e["time_to_success"]
    out = {
        "name": job.name,
        "T_pre": T,
        "seed": job.seed,
        "success": e["success"],
        "time_to_success": t_success,
        "entries": len(runs),
        "left_dead_end": first_exit is not None,
        "went_back": back is not None,
        "reentry_delay": back,
        "recovery": t_success - first_exit * dt
        if t_success is not None and first_exit is not None
        else None,
        "n_marks": e["n_marks"],
    }
    if job.seed == 0:
        out["example"] = e
    return out


def jobs(cfg: dict) -> list[Job]:
    out = []
    for T in T_PRE:
        scenario = {"block_offset": -0.7, "warmup_T": T}
        for name in NAMES:
            lam = None if name == "C2" else cfg["lam"]
            out += [Job(name, M3[name], scenario, lam, s, cfg) for s in range(cfg["seeds"])]
    return out


def tabulate(episodes: list[dict]) -> dict:
    rng = np.random.default_rng(0)

    def ci(values: list) -> list[float]:
        return list(bootstrap_ci(np.array([v for v in values if v is not None], dtype=float), rng))

    table = {}
    for name in NAMES:
        for T in T_PRE:
            eps = [e for e in episodes if e["name"] == name and e["T_pre"] == T]
            left = [e for e in eps if e["left_dead_end"]]
            table[f"{name}@{T:g}"] = {
                "success_rate": ci([e["success"] for e in eps]),
                "time_to_success": ci([e["time_to_success"] for e in eps]),
                "entries": ci([e["entries"] for e in eps]),
                "went_back_if_left": ci([e["went_back"] for e in left]),
                "reentry_delay": ci([e["reentry_delay"] for e in eps]),
                "recovery": ci([e["recovery"] for e in eps]),
                "n_marks": ci([e["n_marks"] for e in eps]),
            }
    return table


def draw_dose(table: dict) -> Figure:
    fig = Figure(figsize=(15, 4.3), layout="constrained")
    fig.suptitle("Right miss with T_pre seconds of the first stage already in the statistic")
    axes = fig.subplots(1, 3)
    keys = (
        ("success_rate", "success within 120 s"),
        ("time_to_success", "time to success (s)"),
        ("reentry_delay", "dead-end exit to sequence start (s)"),
    )
    for colour, name in zip(PALETTE, NAMES, strict=False):
        for ax, (key, _) in zip(axes, keys, strict=True):
            m, lo, hi = np.array([table[f"{name}@{T:g}"][key] for T in T_PRE]).T
            ok = ~np.isnan(m)
            ax.errorbar(
                np.array(T_PRE)[ok],
                m[ok],
                [m[ok] - lo[ok], hi[ok] - m[ok]],
                marker="o",
                capsize=3,
                color=colour,
                label=name,
            )
    for ax, (_, label) in zip(axes, keys, strict=True):
        ax.set(xlabel="warm-up in the first stage, T_pre (s)", ylabel=label, xticks=T_PRE)
        ax.grid(True, color="0.9", lw=0.6)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(frameon=False)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG
    (OUT / "config.json").write_text(
        json.dumps({"config": cfg, "T_pre": T_PRE, "start": START}, indent=2)
    )
    episodes = run_all(jobs(cfg), reentry_outcome)
    examples = {
        e["name"]: e.pop("example") for e in episodes if "example" in e and e["T_pre"] == T_PRE[-1]
    }
    for e in episodes:
        e.pop("example", None)
    (OUT / "episodes.json").write_text(json.dumps(episodes))
    table = tabulate(episodes)
    (OUT / "table.json").write_text(json.dumps(table, indent=2))
    draw_dose(table).savefig(OUT / "reentry_vs_warmup.png", dpi=150)
    T = T_PRE[-1]
    draw_trajectories(
        examples, cfg, f"Seed 0 with {T:g} s of the first stage in the statistic"
    ).savefig(OUT / f"trajectories_warmup{T:g}.png", dpi=150)
    fig = draw_phase(
        examples, cfg, f"Seed 0 with {T:g} s warm-up: phase; grey = dead end, red = firings"
    )
    fig.savefig(OUT / f"phase_warmup{T:g}.png", dpi=150)
    for key, row in table.items():
        print(key, {k: np.round(v, 2).tolist() for k, v in row.items()})


if __name__ == "__main__":
    main()
