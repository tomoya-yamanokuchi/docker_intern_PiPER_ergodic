"""M1: forgetting and the expanding target on the peg_slide task (Scene B), no marks.

Conditions (the statistic, the target):
  B0 full memory, static    B1 hard reset, expanding   C1 exponential, static
  C2 full memory, expanding C3 exponential, expanding  C3f frequency-based, expanding
C1, C3 and C3f are run at every lambda of the sweep; the main table takes lambda = 0.1.

The agent starts where the master starts. The block starts 0.7 short of where it was taught, so
following the master lands the peg on the block's top beside the slot, and it may come down beside
the block: the dead end, where the barrier stops it. A seed sets the demonstrations (so the master
path) and the start's jitter; the plant has no noise. The detector runs but nothing is marked.

Writes results/M1/: config.json, episodes.json, table.json, table.md and the figures.
"""

import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from experiments.sequential_ergodic.episode import CONFIG, Job
from experiments.sequential_ergodic.report import (
    PALETTE,
    draw_phase,
    draw_success,
    draw_trajectories,
    markdown_table,
    row,
    run_all,
)

OUT = Path(__file__).parent / "results" / "M1"
SCENARIO = {"block_offset": -0.7}
LAM_SWEEP = [0.02, 0.05, 0.1, 0.2, 0.5]
CONDITIONS = {
    "B0": {"statistic": "full", "expanding": False},
    "B1": {"statistic": "reset", "expanding": True},
    "C1": {"statistic": "exponential", "expanding": False},
    "C2": {"statistic": "full", "expanding": True},
    "C3": {"statistic": "exponential", "expanding": True},
    "C3f": {"statistic": "frequency", "expanding": True},
}
SWEPT = ("C1", "C3", "C3f")


def jobs(cfg: dict) -> list[Job]:
    out = []
    for name, cond in CONDITIONS.items():
        lams = LAM_SWEEP if name in SWEPT else [None]
        for lam in lams:
            for seed in range(cfg["seeds"]):
                keep = seed == 0 and lam in (None, cfg["lam"])
                out.append(Job(name, cond, SCENARIO, lam, seed, cfg, keep))
    return out


def draw_sweep(sweep: dict) -> Figure:
    fig = Figure(figsize=(15, 4.2), layout="constrained")
    axes = fig.subplots(1, 3)
    keys = ("success_rate", "entries", "revisit_overlap")
    for colour, name in zip(PALETTE[2:], SWEPT, strict=False):
        for ax, key in zip(axes, keys, strict=True):
            m, lo, hi = np.array([sweep[name][lam][key] for lam in LAM_SWEEP]).T
            ax.errorbar(
                LAM_SWEEP, m, [m - lo, hi - m], marker="o", capsize=3, color=colour, label=name
            )
    labels = ("success rate", "dead-end entries per episode", "revisit overlap")
    for ax, label in zip(axes, labels, strict=True):
        ax.set(xscale="log", xlabel=r"$\lambda$ (1/s)", ylabel=label)
        ax.grid(True, color="0.9", lw=0.6)
    axes[0].legend(frameon=False)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG
    (OUT / "config.json").write_text(
        json.dumps({"config": cfg, "scenario": SCENARIO, "lam_sweep": LAM_SWEEP}, indent=2)
    )
    episodes = run_all(jobs(cfg))
    (OUT / "episodes.json").write_text(json.dumps(episodes))
    rng = np.random.default_rng(0)
    main_eps = {
        n: [e for e in episodes if e["name"] == n and e["lam"] in (None, cfg["lam"])]
        for n in CONDITIONS
    }
    sweep = {
        n: {
            lam: row([e for e in episodes if e["name"] == n and e["lam"] == lam], rng)
            for lam in LAM_SWEEP
        }
        for n in SWEPT
    }
    table = {"main": {n: row(eps, rng) for n, eps in main_eps.items()}, "sweep": sweep}
    (OUT / "table.json").write_text(json.dumps(table, indent=2))
    (OUT / "table.md").write_text(markdown_table(table["main"]) + "\n")
    examples = {n: next(e for e in eps if "log" in e) for n, eps in main_eps.items()}
    draw_trajectories(examples, cfg, "M1, seed 0 (world units; red line: master)").savefig(
        OUT / "trajectories_seed0.png", dpi=150
    )
    draw_phase(examples, cfg, "M1, seed 0: phase; grey = in the dead end").savefig(
        OUT / "phase_seed0.png", dpi=150
    )
    draw_success(main_eps, cfg, f"M1, lambda = {cfg['lam']}").savefig(
        OUT / "success_curves.png", dpi=150
    )
    draw_sweep(sweep).savefig(OUT / "lambda_sweep.png", dpi=150)
    print(markdown_table(table["main"]))


if __name__ == "__main__":
    main()
