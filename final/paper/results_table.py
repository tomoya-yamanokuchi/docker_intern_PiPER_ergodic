#!/usr/bin/env python3
"""The door experiment's results table and survival figure for the paper, from the counted run
files. Hardware-free.

Writes results_table.tex beside this file and img/survival.pdf: per condition the trials, the
six metrics of analyse_runs.py, each fraction with its Wilson 95 % interval, and the survival
curve of every counted trial. Then covered_table.tex, the covered peg-in-hole's success rate,
completion time and highest phase per condition.

Run from final/src (the run files name their task relative to it):
    sh run.sh ../paper/results_table.py
"""

from pathlib import Path

import numpy as np

from analyse_runs import plot_survival, trials, wilson_interval

PAPER = Path(__file__).resolve().parent
RUNS = PAPER.parent / "output"
# (name in the paper, method stored in the files, run directories). Listed rather than grouped
# by method, because the single-trial baseline is e2t2 run as one-trial sequences in its own
# directory.
CONDITIONS = [
    ("Full target distribution", "e2t2", ["evaluation", "evaluation_e2t2"]),
    ("Full target distribution (single trial)", "e2t2", ["evaluation_nomemory"]),
    ("Monotonic expansion", "expanding", ["evaluation_expanding"]),
    ("SEC w/o Retreat", "sliding", ["evaluation_sliding"]),
    ("SEC w/o Visitation Decay", "retry_nodecay", ["evaluation_retry_nodecay"]),
    ("SEC", "retry", ["evaluation", "evaluation_retry"]),
]
COVERED_RUNS = PAPER.parent / "saved_recordings" / "evaluation_covered_peg_appendix"
COVERED = [("Full target distribution", "e2t2"), ("SEC", "retry")]


def counted(method: str, dirs: list[str]) -> list[dict]:
    """The counted trials; `_2` keeps retry off retry_nodecay."""
    return [
        t for d in dirs for p in sorted((RUNS / d).glob(f"run_{method}_2*.npz")) for t in trials(p)
    ]


def fraction(k: int, n: int) -> str:
    lo, hi = wilson_interval(k, n)
    # Top-aligned, so the count sits on the condition name's line.
    return rf"\begin{{tabular}}[t]{{@{{}}c@{{}}}}${k}/{n}$\\{k / n:.2f} [{lo:.2f}, {hi:.2f}]\end{{tabular}}"


def row(name: str, ts: list[dict]) -> str:
    n = len(ts)
    wins = [t for t in ts if t["outcome"] == "success"]
    backs = [(t, rec) for t in ts for _, rec in t["setbacks"]]
    recovered = [t["outcome"] == "success" for t, rec in backs if rec]
    with_backs = [t for t in ts if t["setbacks"]]
    q1, med, q3 = np.percentile([t["time"] for t in wins], [25, 50, 75])
    cells = [
        rf"\textbf{{{name}}}" if name == "SEC" else name,
        str(n),
        fraction(len(wins), n),
        fraction(sum(t["outcome"] != "success" for t in ts), n),
        rf"\begin{{tabular}}[t]{{@{{}}c@{{}}}}{med:.0f}\\{{}}[{q1:.0f}, {q3:.0f}]\end{{tabular}}",
        fraction(sum(rec for _, rec in backs), len(backs)),
        fraction(sum(recovered), len(recovered)),
        fraction(sum(t["repeated"] for t in with_backs), len(with_backs)),
    ]
    return " & ".join(cells) + r" \\"


def main() -> None:
    per_condition = {name: counted(method, dirs) for name, method, dirs in CONDITIONS}
    rows = [row(name, ts) for name, ts in per_condition.items()]
    table = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Door task, per condition: count and rate with its Wilson 95\,\% interval for"
            r" each fraction, and the median [IQR] completion time of the successful trials."
            r" Metrics are defined in Table~\ref{tab:metrics}.}",
            r"\label{tab:results}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{>{\raggedright\arraybackslash}p{2.6cm}ccccccc}",
            r"\toprule",
            r"Condition & $n$ & Success & Failure & $t_\mathrm{c}$ (s) & Recovery"
            r" & Post-recovery & Repeated \\",
            r" & & rate & rate & & rate & progress rate & failure rate \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    out = PAPER / "results_table.tex"
    out.write_text(table + "\n")
    plot_survival(per_condition, PAPER / "img" / "survival.pdf")
    print(table)
    print(f"\nwritten to {out} and {PAPER / 'img' / 'survival.pdf'}")
    covered_table()


def covered_trials(method: str) -> list[dict]:
    """The covered peg-in-hole's trials. The phase at insertion lies near 0.8, below the
    success phase of 0.9, so the operator's Enter (logged as an abort) marks a success, and the two
    insertions it missed were corrected to success from the video. Enter comes seconds late,
    so a success is timed when the phase first reaches its trial maximum."""
    found = []
    for p in sorted(COVERED_RUNS.glob(f"run_{method}_2*.npz")):
        run = np.load(p)
        for k, outcome in enumerate(run["outcomes"]):
            mine = run["trial"] == k
            t, phi = run["t"][mine], run["phi"][mine]
            found.append(
                {
                    "success": outcome in ("success", "abort"),
                    "time": float(t[phi.argmax()]),
                    "peak": float(phi.max()),
                }
            )
    return found


def covered_row(name: str, ts: list[dict]) -> str:
    times = [t["time"] for t in ts if t["success"]]
    peaks = [t["peak"] for t in ts]
    time = "--"
    if times:
        q1, med, q3 = np.percentile(times, [25, 50, 75])
        time = rf"{med:.0f} [{q1:.0f}, {q3:.0f}]"
    cells = [
        rf"\textbf{{{name}}}" if name == "SEC" else name,
        str(len(ts)),
        fraction(len(times), len(ts)),
        time,
        rf"{np.median(peaks):.2f} [{min(peaks):.2f}, {max(peaks):.2f}]",
    ]
    return " & ".join(cells) + r" \\"


def covered_table() -> None:
    rows = [covered_row(name, covered_trials(method)) for name, method in COVERED]
    table = "\n".join(
        [
            r"\begin{table}[t]",
            r"\caption{Covered peg-in-hole, per condition: successes with the Wilson 95\,\%"
            r" interval, the median [IQR] completion time of the successful trials, and the"
            r" median [range] of the highest phase each trial reached.}",
            r"\label{tab:covered}",
            r"\begin{center}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{>{\raggedright\arraybackslash}p{1.7cm}cccc}",
            r"\toprule",
            r"Condition & $n$ & Success rate & $t_\mathrm{c}$ (s) & Highest phase \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            r"\end{table}",
        ]
    )
    out = PAPER / "covered_table.tex"
    out.write_text(table + "\n")
    print(table)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
