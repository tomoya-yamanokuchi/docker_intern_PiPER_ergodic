#!/usr/bin/env python3
"""The experiment's metrics over finished runs, per method. Hardware-free.

Every metric is read from the logged phase, the one signal every method has; GLOSSARY
defines setback, recovery and time to unlock. A trial counts if it ended in success,
limit or abort, the last two failures; error and interrupted are the rig's, not the
method's, and are left out. A run file is one sequence, so a trial's index in it is its
index in the sequence.

  success rate              successes / counted trials
  failure rate, by cause    limits / counted trials, aborts / counted trials
  completion time           s to success, successes only: median and IQR; the survival
                            curve of every counted trial goes to survival.png
  time to unlock            s, successes only, secondary
  recovery rate             recovered setbacks / setbacks
  post-recovery progress    recovered setbacks in a successful trial / recovered setbacks
  repeated-failure rate     trials with two setbacks whose peaks lie within one resolvable
                            width / trials with a setback
  resets                    failed walks / walks

Fractions carry a Wilson 95 % interval.

Run from final/src:  sh run.sh analyse_runs.py run_<method>_<ts>.npz ...
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from method.task import Task

COUNTED = ("success", "limit", "abort")
TURN_END = 0.50  # phase; on the door master the key has turned 43 of its 47 deg by here
TIME_LIMIT = 600.0  # s, a trial's on the arm
Z = 1.96  # 95 %


def wilson(k: int, n: int) -> str:
    if n == 0:
        return "-"
    p, z2 = k / n, Z**2 / n
    centre = (p + z2 / 2) / (1 + z2)
    half = Z * np.sqrt(p * (1 - p) / n + z2 / (4 * n)) / (1 + z2)
    return f"{k}/{n} = {p:.2f}  [{centre - half:.2f}, {centre + half:.2f}]"


def spread(values: list[float]) -> str:
    if not values:
        return "-"
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return f"median {med:.1f} s  IQR [{q1:.1f}, {q3:.1f}] s  (n = {len(values)})"


def setbacks(phi: np.ndarray, width: float) -> list[tuple[float, bool]]:
    """(peak, recovered) for every setback in one trial's phase.

    After a setback the next peak starts only at an advance, so one long fall is one setback.
    """
    found, peak, trough = [], -np.inf, None
    for k, p in enumerate(phi):
        if trough is None:
            peak = max(peak, p)
            if p < peak - width:
                found.append((peak, k))
                trough = p
        elif p >= (trough := min(trough, p)) + width:
            peak, trough = p, None
    return [(pk, bool((phi[k:] > pk).any())) for pk, k in found]


def unlock_time(t: np.ndarray, phi: np.ndarray) -> float | None:
    """When the phase last rose past TURN_END, staying past it to the end; None if it ends below."""
    above = phi > TURN_END
    if not above[-1]:
        return None
    below = np.flatnonzero(~above)
    return float(t[below[-1] + 1] if len(below) else t[0])


def trials(path: Path) -> list[dict]:
    run = np.load(path)
    width = Task.load(Path(str(run["task"]))).sigma_f
    found = []
    for k, outcome in enumerate(run["outcomes"]):
        mine = run["trial"] == k
        if outcome not in COUNTED or not mine.any():  # an abort before the first law step
            continue
        t, phi = run["t"][mine], run["phi"][mine]
        backs = setbacks(phi, width)
        peaks = np.array([pk for pk, _ in backs])
        found.append(
            {
                "index": k,
                "outcome": str(outcome),
                "time": float(t[-1]),
                "unlock": unlock_time(t, phi) if outcome == "success" else None,
                "setbacks": backs,
                "repeated": bool((np.abs(peaks[:, None] - peaks[None]) < width).sum() > len(peaks)),
            }
        )
    return found


def report(method: str, ts: list[dict], walks: np.ndarray) -> None:
    n, wins = len(ts), [t for t in ts if t["outcome"] == "success"]
    backs = [(t, s) for t in ts for s in t["setbacks"]]
    recovered = [t["outcome"] == "success" for t, (_, rec) in backs if rec]
    with_backs = [t for t in ts if t["setbacks"]]
    print(f"\n{method}: {n} trials")
    print(f"  success rate            {wilson(len(wins), n)}")
    for cause in COUNTED[1:]:
        print(f"  failure rate, {cause:5s}     {wilson(sum(t['outcome'] == cause for t in ts), n)}")
    print(f"  completion time         {spread([t['time'] for t in wins])}")
    print(
        f"  time to unlock          {spread([t['unlock'] for t in wins if t['unlock'] is not None])}"
    )
    print(f"  recovery rate           {wilson(sum(rec for _, (_, rec) in backs), len(backs))}")
    print(f"  post-recovery progress  {wilson(sum(recovered), len(recovered))}")
    print(
        f"  repeated-failure rate   {wilson(sum(t['repeated'] for t in with_backs), len(with_backs))}"
    )
    by_index = defaultdict(list)
    for t in ts:
        by_index[t["index"]].append(t["outcome"] == "success")
    for i, wins_i in sorted(by_index.items()):
        print(f"  trial {i + 1} of a sequence  {wilson(sum(wins_i), len(wins_i))}")
    print(
        f"  resets failed           {int((walks[:, 1] == 0).sum()) if len(walks) else 0}/{len(walks)}"
    )


def plot_survival(per_method: dict, path: Path) -> None:
    """The fraction of trials not yet a success, against time, per method."""
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot()
    for method, ts in per_method.items():
        times = np.sort([t["time"] for t in ts if t["outcome"] == "success"])
        left = 1 - np.arange(len(times) + 1) / len(ts)
        ax.step(np.r_[0.0, times, TIME_LIMIT], np.r_[left, left[-1]], where="post", label=method)
    ax.set(xlabel="t (s)", ylabel="trials not yet a success", ylim=(0, 1.02), xlim=(0, TIME_LIMIT))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)


def main(paths: list[Path]) -> None:
    per_method, walks = defaultdict(list), defaultdict(list)
    for path in paths:
        run = np.load(path)
        per_method[str(run["method"])] += trials(path)
        if "reset_done" in run.files:
            walks[str(run["method"])] += list(run["reset_done"])
    for method, ts in per_method.items():
        report(method, ts, np.array(walks[method]).reshape(-1, 3))
    plot = paths[0].with_name("survival.png")
    plot_survival(per_method, plot)
    print(f"\nsurvival curve in {plot}")


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
