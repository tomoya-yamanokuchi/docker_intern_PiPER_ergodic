#!/usr/bin/env python3
"""The door experiment's results table for the paper, from the counted run files. Hardware-free.

Writes results_table.tex beside this file: per condition the trials, the success rate with its
Wilson 95 % interval, Fisher's exact test (two-sided, uncorrected) against Ours, and the
completion time of the successes.

Run from final/src (the run files name their task relative to it):
    sh run.sh ../paper/results_table.py
"""

from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

from analyse_runs import TIME_LIMIT, trials, wilson_interval

RUNS = Path(__file__).resolve().parents[1] / "output"
# (name in the paper, method stored in the files, run directories). Listed rather than grouped
# by method, because E2T2 w/o memory is e2t2 run as one-trial sequences in its own directory.
CONDITIONS = [
    ("Ours", "retry", ["evaluation", "evaluation_retry"]),
    ("Ours w/o Retreat", "sliding", ["evaluation_sliding"]),
    ("Ours w/o Visitation Decay", "retry_nodecay", ["evaluation_retry_nodecay"]),
    ("E2T2", "e2t2", ["evaluation", "evaluation_e2t2"]),
    ("E2T2 w/o memory", "e2t2", ["evaluation_nomemory"]),
    ("Monotonic expansion", "expanding", ["evaluation_expanding"]),
]


def successes(method: str, dirs: list[str]) -> tuple[int, list[float]]:
    """(counted trials, completion times in s of the successes); `_2` keeps retry off retry_nodecay."""
    ts = [
        t for d in dirs for p in sorted((RUNS / d).glob(f"run_{method}_2*.npz")) for t in trials(p)
    ]
    return len(ts), [t["time"] for t in ts if t["outcome"] == "success"]


def p_value(p: float) -> str:
    if p >= 1e-3:
        return f"{p:.3f}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def row(name: str, n: int, times: list[float], ref: tuple[int, int] | None) -> str:
    k = len(times)
    lo, hi = wilson_interval(k, n)
    p = "--" if ref is None else p_value(fisher_exact([list(ref), [k, n - k]])[1])
    q1, med, q3 = np.percentile(times, [25, 50, 75])
    return (
        rf"{name} & {n} & {k / n:.2f} [{lo:.2f}, {hi:.2f}] & {p} & "
        rf"{med:.0f} [{q1:.0f}, {q3:.0f}] \\"
    )


def main() -> None:
    rows, ref = [], None
    for name, method, dirs in CONDITIONS:
        n, times = successes(method, dirs)
        rows.append(row(name, n, times, ref))
        if ref is None:  # the first condition is Ours
            ref = (len(times), n - len(times))
    table = "\n".join(
        [
            r"\begin{table*}[t]",  # both columns: the table is about 400 pt wide, a column 252 pt
            r"\centering",
            r"\caption{Door task, per condition: success rate with its Wilson 95\,\% interval,"
            r" Fisher's exact test of the success rate against Ours (two-sided, uncorrected), and"
            r" completion time of the successful trials. A trial fails at the"
            rf" {TIME_LIMIT:.0f}\,s limit.}}",
            r"\label{tab:results}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Condition & $n$ & Success [95\,\% CI] & $p$ vs Ours & $t_\mathrm{c}$ (s), median [IQR] \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    out = Path(__file__).with_name("results_table.tex")
    out.write_text(table + "\n")
    print(table)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
