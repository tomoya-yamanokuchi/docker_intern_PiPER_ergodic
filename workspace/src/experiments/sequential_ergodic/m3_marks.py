"""M3: phantom-mass marks on the peg_slide right miss (block 0.7 short), expanding target.

Conditions, with exponential forgetting at lambda = 0.1 (and an f suffix: frequency-based):
  C3  no marks (the detector runs; marks are recorded only)
  C4  inside: the mark into the statistic, forgotten like a visit
  C5  separate, phase-conditioned; C5g the same, global
  C6a zeroed target components under the mark;  C6b the mark as a negative target term
C2 (full memory, no marks) is the reference from M1. Leakage (P4): K in {6, 16, 40} for C3 and C5.

Writes results/M3/: config.json, episodes.json, table.json, table.md and the figures.
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
    draw_world,
    markdown_table,
    row,
    run_all,
)
from simulation.peg_slide import C0, SIZE

OUT = Path(__file__).parent / "results" / "M3"
SCENARIO = {"block_offset": -0.7}
K_SWEEP = [6, 16, 40]
_EXP = {"statistic": "exponential", "expanding": True}
_FREQ = {"statistic": "frequency", "expanding": True}
VARIANTS = {
    "3": None,
    "4": "inside",
    "5": "separate",
    "5g": "separate_global",
    "6a": "zeroed",
    "6b": "negative",
}
CONDITIONS = {"C2": {"statistic": "full", "expanding": True}}
CONDITIONS |= {f"C{k}": _EXP | {"marks": v} for k, v in VARIANTS.items()}
CONDITIONS |= {f"C{k}f": _FREQ | {"marks": v} for k, v in VARIANTS.items()}
EXPONENTIAL = ["C2", *(f"C{k}" for k in VARIANTS)]
FREQUENCY = ["C2", *(f"C{k}f" for k in VARIANTS)]


def jobs(cfg: dict) -> list[Job]:
    out = []
    for name, cond in CONDITIONS.items():
        lam = None if name == "C2" else cfg["lam"]
        out += [Job(name, cond, SCENARIO, lam, s, cfg, s == 0) for s in range(cfg["seeds"])]
    for K in K_SWEEP:
        if cfg["K"][0] == K:
            continue  # the main runs are K = 16
        kcfg = cfg | {"K": [K, K]}
        for name in ("C3", "C5"):
            out += [
                Job(f"{name}@K{K}", CONDITIONS[name], SCENARIO, cfg["lam"], s, kcfg)
                for s in range(cfg["seeds"])
            ]
    return out


def reentry_intervals(eps: list[dict]) -> list[float]:
    """Seconds between successive dead-end entries after the first mark."""
    out = []
    for e in eps:
        t0 = e["marks"][0][4] if e["marks"] else np.inf
        later = [t for t in e["entry_times"] if t > t0]
        out += list(np.diff(later))
    return out


def draw_heatmaps(groups: dict[str, list[dict]], names: list[str]) -> Figure:
    """Where the mark mass lies, summed over all seeds: mass x shape weight per 0.2-unit cell."""
    fig = Figure(figsize=(4 * len(names), 4), layout="constrained")
    fig.suptitle(
        "Mark mass over all 30 seeds (the block drawn where it starts; its slot is the real hole)"
    )
    axes = fig.subplots(1, len(names))
    edges = np.linspace(0, SIZE, 51)
    for ax, name in zip(axes, names, strict=True):
        H = np.zeros((50, 50))
        for e in groups[name]:
            for points, weights, mass, _, _ in e["marks"]:
                p = np.array(points) * SIZE
                H += np.histogram2d(
                    p[:, 0], p[:, 1], [edges, edges], weights=mass * np.array(weights)
                )[0]
        draw_world(ax, C0 + SCENARIO["block_offset"], True)
        ax.pcolormesh(edges, edges, np.ma.masked_equal(H.T, 0), cmap="Reds", zorder=3)
        ax.set_title(f"{name}: total mass {H.sum():.0f} s", fontsize=10)
    return fig


def draw_k_sweep(table: dict) -> Figure:
    fig = Figure(figsize=(10, 4), layout="constrained")
    axes = fig.subplots(1, 2)
    for colour, name in zip(PALETTE[1:], ("C3", "C5"), strict=False):
        rows = [table["k_sweep"][name][str(K)] for K in K_SWEEP]
        for ax, key in zip(axes, ("success_rate", "false_mark_rate"), strict=True):
            m, lo, hi = np.array([r[key] for r in rows]).T
            ax.errorbar(
                K_SWEEP, m, [m - lo, hi - m], marker="o", capsize=3, color=colour, label=name
            )
    for ax, label in zip(axes, ("success rate", "false-mark rate"), strict=True):
        ax.set(
            xscale="log",
            xticks=K_SWEEP,
            xticklabels=K_SWEEP,
            xlabel="K (modes per axis)",
            ylabel=label,
        )
        ax.grid(True, color="0.9", lw=0.6)
    axes[0].legend(frameon=False)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG
    (OUT / "config.json").write_text(
        json.dumps({"config": cfg, "scenario": SCENARIO, "K_sweep": K_SWEEP}, indent=2)
    )
    episodes = run_all(jobs(cfg))
    (OUT / "episodes.json").write_text(
        json.dumps([{k: v for k, v in e.items() if k != "log"} for e in episodes])
    )
    rng = np.random.default_rng(0)
    groups = {
        n: [e for e in episodes if e["name"] == n]
        for n in dict.fromkeys(e["name"] for e in episodes)
    }
    table = {"main": {n: row(groups[n], rng) for n in CONDITIONS}}
    for n in CONDITIONS:
        intervals = reentry_intervals(groups[n])
        table["main"][n]["reentry_interval"] = [
            float(np.mean(intervals)) if intervals else None,
            len(intervals),
        ]
    table["k_sweep"] = {
        n: {str(K): row(groups[n if cfg["K"][0] == K else f"{n}@K{K}"], rng) for K in K_SWEEP}
        for n in ("C3", "C5")
    }
    (OUT / "table.json").write_text(json.dumps(table, indent=2))
    (OUT / "table.md").write_text(markdown_table(table["main"]) + "\n")
    examples = {n: next(e for e in groups[n] if "log" in e) for n in EXPONENTIAL}
    draw_trajectories(examples, cfg, "M3, seed 0, exponential forgetting (red: marks)").savefig(
        OUT / "trajectories_seed0.png", dpi=150
    )
    draw_phase(
        examples, cfg, "M3, seed 0: phase; red lines = firings, shaded = marked segment"
    ).savefig(OUT / "phase_seed0.png", dpi=150)
    draw_success({n: groups[n] for n in EXPONENTIAL}, cfg, "M3, exponential forgetting").savefig(
        OUT / "success_exponential.png", dpi=150
    )
    draw_success({n: groups[n] for n in FREQUENCY}, cfg, "M3, frequency-based forgetting").savefig(
        OUT / "success_frequency.png", dpi=150
    )
    draw_heatmaps(groups, ["C4", "C5", "C6a", "C6b"]).savefig(OUT / "mark_heatmaps.png", dpi=150)
    draw_k_sweep(table).savefig(OUT / "k_sweep.png", dpi=150)
    print(markdown_table(table["main"]))
    print({n: table["main"][n]["reentry_interval"] for n in CONDITIONS})


if __name__ == "__main__":
    main()
