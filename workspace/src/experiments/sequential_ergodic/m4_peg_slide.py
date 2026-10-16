"""M4: phase conditioning of the marks on the peg_slide task, over three failures.

Scenarios: right_miss (block 0.7 short: the peg comes down beside it and runs into the barrier),
left_miss (block 0.7 long: a control, the agent finds the slot without a dead end), incomplete_push
(block where taught; a scripted peg pushes it 40 % of the way and lifts out; the controller starts
above the moved block with an empty statistic).
Conditions, exponential forgetting at lambda = 0.1, expanding target:
  C3 no marks; C5 separate, phase-conditioned; C5g separate, global; C5s C5 with phi_uses_slider_state;
  C5p (a deviation from the spec) C5 with marks tagged at phi_tilde's peak instead of at firing.
C2 (full memory, no marks) is the reference.

Writes results/M4/: config.json, episodes.json, table.json, table.md and the figures.
"""

import json
from pathlib import Path

import numpy as np

from experiments.sequential_ergodic.episode import CONFIG, Job
from experiments.sequential_ergodic.report import (
    draw_success,
    draw_trajectories,
    markdown_table,
    row,
    run_all,
)

OUT = Path(__file__).parent / "results" / "M4"
SCENARIOS = {
    "right_miss": {"block_offset": -0.7},
    "left_miss": {"block_offset": 0.7},
    "incomplete_push": {"block_offset": 0.0, "prefix": "incomplete_push", "push_fraction": 0.4},
}
_C5 = {"statistic": "exponential", "expanding": True, "marks": "separate"}
CONDITIONS = {
    "C2": {"statistic": "full", "expanding": True},
    "C3": {"statistic": "exponential", "expanding": True},
    "C5": _C5,
    "C5g": _C5 | {"marks": "separate_global"},
    "C5s": _C5 | {"slider_phase": True},
    "C5p": _C5 | {"mark_params": {"tag_at_peak": True}},
}
COLUMNS = {
    "success_rate": ("success", ".2f"),
    "time_to_success": ("time to success (s)", ".1f"),
    "entries": ("dead-end entries", ".1f"),
    "n_marks": ("marks", ".1f"),
    "marks_at_hole": ("marks at the hole", ".2f"),
    "false_mark_rate": ("false-mark rate", ".2f"),
    "dead_end_mark_share": ("mark mass in dead end", ".2f"),
}


def jobs(cfg: dict) -> list[Job]:
    out = []
    for scen_name, scen in SCENARIOS.items():
        for name, cond in CONDITIONS.items():
            lam = None if name == "C2" else cfg["lam"]
            out += [
                Job(f"{scen_name}/{name}", cond, scen, lam, s, cfg, s == 0)
                for s in range(cfg["seeds"])
            ]
    return out


def hole_marks_vs_success(eps: list[dict]) -> dict:
    """Success with and without a mark on the way into the real slot: did a mark block it?"""
    with_mark = [e["success"] for e in eps if e["marks_at_hole"]]
    without = [e["success"] for e in eps if not e["marks_at_hole"]]
    return {
        "episodes_with_mark_at_hole": len(with_mark),
        "success_with": float(np.mean(with_mark)) if with_mark else None,
        "success_without": float(np.mean(without)) if without else None,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG
    (OUT / "config.json").write_text(json.dumps({"config": cfg, "scenarios": SCENARIOS}, indent=2))
    episodes = run_all(jobs(cfg))
    (OUT / "episodes.json").write_text(
        json.dumps([{k: v for k, v in e.items() if k != "log"} for e in episodes])
    )
    rng = np.random.default_rng(0)
    table, md = {}, []
    for scen in SCENARIOS:
        groups = {n: [e for e in episodes if e["name"] == f"{scen}/{n}"] for n in CONDITIONS}
        table[scen] = {
            n: row(eps, rng) | {"hole": hole_marks_vs_success(eps)} for n, eps in groups.items()
        }
        md += [f"### {scen}", "", markdown_table(table[scen], COLUMNS), ""]
        examples = {n: next(e for e in eps if "log" in e) for n, eps in groups.items()}
        draw_trajectories(examples, cfg, f"M4 {scen}, seed 0 (red: marks)").savefig(
            OUT / f"trajectories_{scen}.png", dpi=150
        )
        draw_success(groups, cfg, f"M4 {scen}").savefig(OUT / f"success_{scen}.png", dpi=150)
    (OUT / "table.json").write_text(json.dumps(table, indent=2))
    (OUT / "table.md").write_text("\n".join(md))
    print("\n".join(md))
    print(json.dumps({s: {n: r["hole"] for n, r in t.items()} for s, t in table.items()}, indent=1))


if __name__ == "__main__":
    main()
