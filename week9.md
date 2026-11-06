# Week 9: comparison and ablation experiments

Brief for an agent working this out. Settled in a grilling session on 2026-11-02; nothing below is open except the last section.

## Goal

Run multiple experiments comparing our method with its baselines, and an ablation study showing which component of the approach does the largest work. Supervisor's brief (Week 2 of the paper plan): specify the experiments, define each condition and each metric operationally, and draft the Experiments section with placeholders for results not yet available.

## Conditions

Five conditions run on the arm in the door task (`final/`). The 2-D sim is supporting material only.

| Supervisor's name | Code (`METHODS` in `final/src/method/law.py`) | Trials |
|---|---|---|
| Full target distribution | `e2t2` | 48 |
| Monotonic expansion | `expanding` | 24 |
| Ours w/o Retreat | `sliding` (sliding target + forgetting, no retreat) | 24 |
| Ours w/o Visitation Decay | `retry` with forgetting factor `lam = 1` (`retry_nodecay` in `METHODS`, added uncommitted) | 24 |
| Ours | `retry` (sliding target + forgetting + retreat) | 48 |
| E2T2 w/o memory (sixth) | `e2t2` run as independent one-trial runs, so the statistic is always empty; 13 exist from 10-30, 11 to run | 24 |

- "Persistent Retreat Memory" (the supervisor's tentative term) is the parked stall-memory approach (`park/stall-memory`). It was tested and rejected on the arm (memory 4/8, retry 7/8), so it is not a condition. The supervisor's ablation is renamed "Ours w/o Retreat". Tell the supervisor why.
- "Visitation decay" is the statistic's forgetting factor `lam = exp(-dt / T_f)`.
- Framing for Ours: a sliding target that forgets where it has been and, on a stall, retreats its back edge along the master at the arm's own speed, so earlier stages return to the target.

## Protocol

- Roster: 48 trials = 12 sequences of 4 for `e2t2` and `retry` (frozen, already partly run: 28 e2t2 and 9 retry files in `final/output/evaluation/`). 24 trials = 6 sequences of 4 for each of the other three conditions. Fixed n, no early stop; state the smaller n as a limitation.
- Sequence is the sample unit. `e2t2` hands its statistic across the trials of a sequence; the forgetting methods start every trial fresh (GLOSSARY "Sequence").
- Success = phase > 0.9 within 600 s. Automatic reset after every trial (`--auto`). Session order: alternate the starting method per session, as in the existing roster, unless told otherwise.
- Demonstration: master and exploration recordings in `final/saved_recordings/door/`; task frozen as `task_20261023_131342` (tag `door-experiment-v1`). Describe how they were collected from `final/CLAUDE.md` and the glossary.
- Code: `door-experiment-v4.1` is the frozen code of the e2t2 and retry runs already taken. The working tree is ahead of it, uncommitted in `final/src/method/law.py`: the `expanding` method and its `PhasedStatistic` (needed for the Monotonic expansion condition, absent from v4.1), the `retry_nodecay` entry, and a `new_trial` change (`self.target == "sliding"` instead of `self.lam < 1.0`, so `retry_nodecay` also gets a fresh statistic each trial; the existing methods take the same branch as before). Other uncommitted edits sit in `run_trials.py`, `simulate.py` and `visualization/`. Before the new conditions run, the user commits these and tags `door-experiment-v4.3` (v4.2 is the same without the `--out` flag); the agent checks with `test_against_notebook` and a diff against v4.1 that the e2t2 and retry paths are unchanged, so the existing runs stay pooled. Do not commit or tag unless asked.
- Never run an ARM script (see the root `CLAUDE.md` hard rule 1). The user runs the arm; use `simulate.py` and the fake arm to check offline.

## Metrics

Already computed by `final/src/analyse_runs.py` (`sh run.sh analyse_runs.py ../output/evaluation/run_*.npz`). Write each out in the report as numerator, denominator and unit, from that file and the GLOSSARY ("Success", "Setback", "Time to unlock"):

- Task success rate
- Task completion time (time to unlock)
- Task failure rate (by cause: limit, abort)
- Recovery rate
- Post-recovery progress rate
- Repeated-failure rate

Ablation ranking: success-rate drop from Ours per ablation, Wilson 95 % intervals; the other five metrics descriptive. State plainly when intervals overlap (about +/-20 points at n = 24).

## Report work (`weekly_reports/9/`, skill `weekly-report`)

The directory exists and compiles; `Content.tex` is empty and the Progress and Goals bullets are blank.

1. Progress and Goals bullets: the user's words. Goal sentence: "run multiple experiments to compare our methodology and perform an ablation study to see what component of the approach does the largest work." Ask before writing anything else into Progress.
2. Outline `Content.tex`: one subsection for the experimental conditions, one for the metric definitions, in the skill's outline format. No invented numbers; unmeasured values are `\textbf{[TODO: ...]}`.
3. Compile with the latexmk command in the skill and list the TODOs.

## Open

- Session order for the three new conditions (assumed: alternating start, as the existing roster).
- No between-method test exists yet in `analyse_runs.py` (Fisher / log-rank); it was offered earlier and not asked for.
