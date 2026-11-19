# Report edits: door top-up and covered peg-in-hole

A step-by-step guide for editing the report on another computer. It is written against
`final/paper/manuscript.tex` as of 2026-11-09 13:47 (the newest copy in the repo, which already
holds the 11-09 changes such as "Sequential Ergodic Control", "retreat" and "expert trajectory").
If Overleaf has moved on since, find each place by its section heading and the quoted sentence.

Every edit says **when** it can be made:

- **NOW**: needs no new results, write it today.
- **AFTER PEG**: after the covered peg-in-hole trials (2 x 5 full target distribution, 10 Ours).
- **AFTER DOOR**: after door session 12 and a rerun of `final/paper/results_table.py`.

`XX` marks a number that is not known yet. Never fill a door number by hand: it comes from the
`results_table.py` output. The covered peg numbers come from the small table function described
in step 13.

---

## Fixed decisions (so you don't have to re-decide while editing)

- Door: all six conditions at 48 trials. Sessions 7-12 add 6 sequences each of monotonic
  expansion, Ours w/o Retreat and Ours w/o Visitation Decay, and 24 single trials of the full
  target distribution. Same arm (S-V1.8-6), same code (`door-experiment-v4.4`), same door rig:
  the environment was not changed between the halves (user, 2026-11-13). The paper mentions no
  rebuild and no rig-check sequence.
- Covered peg: full target distribution **with memory**, 2 sequences of 5, against Ours, 10
  trials. The other PiPER arm (S-V1.8-2 firmware).
- Covered peg success: the operator confirms once the peg is in the hole; a trial at the 600 s
  limit is a failure. Success time = first time the phase reaches its trial maximum. The two
  outcomes corrected from video are not mentioned in the report (user, 2026-11-13).
- Covered peg wording, decided before the results:
  - intervals do not overlap: "Ours succeeded more often than the full target distribution";
  - intervals overlap: "Ours succeeded in XX of 10 trials and the full target distribution in
    XX of 10; with 10 trials per condition the difference is not conclusive".
- Names as in `final/GLOSSARY.md`. No p-values: Wilson 95 % intervals, "different" only when
  they do not overlap.

## Numbers already verified (computed 2026-11-12 from the task files)

| | Door (`task_20261023_131342`) | Covered peg (`task_20261111_194118`) |
|---|---|---|
| N_m (intervals) | 400 | 400 |
| S_m | 0.320 m | 0.305 m |
| N_d | 167 | 133 |
| h | 1.96 mm | 2.15 mm |
| delta | 0.060 | 0.038 |
| datapoints per tenth of phase | 8, 8, 30, 73, 21, 8, 4, 5, 4, 6 | 8, 4, 6, 14, 29, 13, 8, 8, 13, 30 |

The door column matches the manuscript exactly, which checks the computation. Wilson half-width
at a rate of 0.5: about 26 points at n = 10, 19 at n = 24, 14 at n = 48.

---

## 1. Abstract — NOW, one number AFTER DOOR

Replace

> On a locked-door task with a real AgileX PiPER arm, our method reached the highest success rate
> of six conditions, 85\,\% of 48 trials. An ablation shows that the retreat does the most work:
> without it, success falls to 12\,\%.

with

```latex
On a locked-door task with a real AgileX PiPER arm, our method reached the highest success rate
of six conditions, 85\,\% of 48 trials. An ablation shows that the retreat does the most work:
without it, success falls to XX\,\%. On a second task, a peg-in-hole under a hinged cover, it
succeeded in XX of 10 trials against XX of 10 for ergodic control over the full target
distribution.
```

The 85 % of 48 stays (Ours gets no new trials). The 12 % changes after the top-up. If the covered
peg intervals overlap, keep the counts but add nothing that reads as a ranking.

## 2. Introduction, contributions — NOW

Replace the third bullet

> \item an evaluation on a sequential locked-door task on a real AgileX PiPER arm, with an
> ablation of each component.

with

```latex
\item an evaluation on two sequential tasks with real AgileX PiPER arms: a locked door, with an
ablation of each component, and a peg-in-hole under a hinged cover.
```

## 3. Problem formulation — NOW

The section opens with "The task considered is opening a locked door with a key held by a robot
arm." Keep the door as the running example, but add after the three-subtask sentence:

```latex
The formulation does not depend on the door: Sec.~\ref{sec:covered} applies it unchanged to a
second task, a peg-in-hole under a hinged cover, with two subtasks.
```

## 4. Experiments, opening paragraph — NOW

Replace "The experiments ask two questions." and the two questions with:

```latex
The experiments ask three questions. Does Sequential Ergodic Control complete the sequential
locked-door task more reliably than ergodic control over the full target distribution and than a
target that only grows with progress? Which of its two components, the retreat of
Sec.~\ref{sec:retry} or the visitation decay of Sec.~\ref{sec:statistic}, does the most work?
And does it carry over to a second sequential task, a covered peg-in-hole
(Sec.~\ref{sec:covered})?
```

Keep the sentence about the taught lock pose after it.

## 5. Conditions table (`tab:conditions`) — DONE 2026-11-13 in the repo manuscript

Trials column: every 4-trial row becomes `$12\times4$`, the single-trial row `$48\times1$`.

## 6. Protocol (door) — text DONE 2026-11-13 in the repo manuscript, XX filled AFTER DOOR

Replace in the third paragraph

> The number of trials was fixed in advance and not stopped early: 12 sequences for the full
> target distribution and for Ours, 6 for each other condition, and 24 single trials. The runs
> took place in eight sessions over six days.

with

```latex
The number of trials was fixed in advance and not stopped early: 12 sequences for every
condition and 48 single trials. The runs took place in XX sessions over XX days.
```

In the second paragraph, recount "(10 of 264 resets: 3 for the full target distribution, 2 for
monotonic expansion, 5 for Ours w/o Retreat)" from the Notes columns of the session tables in
`experiment_roster.md` (sessions 1-6 and 7-12). Every reset that hit the 60 s limit counts.

(Superseded 2026-11-13: no rebuild paragraph; the door rig was unchanged.)

Check against the new files that "no trial was aborted" is still true.

## 7. Metrics — DONE 2026-11-13 in the repo manuscript

Replace "with 24 trials an interval spans about $\pm20$ points." with

```latex
with 48 trials an interval spans about $\pm14$ points at a rate of 0.5.
```

## 8. Results (door) — AFTER DOOR; branch wordings are `%` comments in the repo manuscript

1. Run `results_table.py`, replace the table with the new `results_table.tex`, upload the new
   `img/survival.pdf`.
2. The full target distribution and Ours rows do not change; everything else does.
3. Rewrite from the new table: *Success* (which intervals overlap Ours), *Ablation* (both drops,
   now with 48 trials each, and whether the visitation decay's interval still overlaps),
   *Completion time*, *Recovery*. Do not keep any old sentence unchecked.
4. *Trials within a sequence*: the per-index 4/12, 9/12, 10/12, 10/12 stay; the single-trial
   result becomes XX of 48.

## 9. New subsection: covered peg-in-hole — DONE 2026-11-13 in `final/paper/manuscript.tex`

Written into the repo manuscript with the final numbers (6/10 vs 0/10), Table V inlined from
`final/paper/covered_table.tex`, and the abstract, contributions, problem formulation,
Experiments opening, Discussion (fifth limitation) and Conclusions edits (1-4, 10, 11 for the
covered peg). Copy those passages to Overleaf. The drafts below are superseded by it.

Put it after `\subsection{Results}` and before `\subsection{Discussion}`. Drafts:

```latex
\subsection{Covered peg-in-hole}
\label{sec:covered}
The second task inserts a peg into a hole under a hinged cover (Fig.~\ref{fig:covered}). The
peg must first lift the cover and then enter the hole. The cover is held shut by gravity and
falls back as soon as the peg moves away from the hinge, so an insertion attempt that strays
undoes the opening, and the two subtasks again succeed only in order.

The demonstration is collected as for the door: one expert trajectory, resampled into
$N_m=400$ intervals of total length $S_m=0.305$~m, and $N_d=133$ exploration datapoints; per
tenth of phase they number 8, 4, 6, 14, 29, 13, 8, 8, 13 and 30. The kernel width is
$h=2.15$~mm and the resolvable width $\delta=0.038$. Because the cover does not close when the
peg is withdrawn, the reset of Sec.~\ref{sec:protocol} adds a taught closing trajectory after
playing the expert trajectory back.

We compare Ours with the full target distribution, the strongest baseline on the door, in 2
sequences of 5 trials with its statistic handed on within a sequence, against 10 trials of Ours.
The laws, gains and time limit are those of the door; the arm is a second PiPER of the same
model. The phase at which the peg sits in the hole is about 0.8, below the door's success
threshold of 0.9, so success is confirmed by the operator once the peg is in the hole, and its
time is the first time the phase reaches its maximum in that trial.
```

Results paragraph, AFTER PEG, one of the two fixed wordings:

```latex
Ours succeeded in XX of 10 trials (XX [XX, XX]) and the full target distribution in XX of 10
(XX [XX, XX]), with median completion times of XX~s and XX~s (Table~\ref{tab:covered}).
```

then either "Ours succeeded more often." (intervals apart) or "With 10 trials per condition the
difference is not conclusive." (intervals overlap).

Table:

```latex
\begin{table}[t]
\caption{Covered peg-in-hole, per condition: successes with the Wilson 95\,\% interval, and the
median [IQR] completion time of the successful trials.}
\label{tab:covered}
\begin{center}
\footnotesize
\begin{tabular}{lccc}
\toprule
Condition & $n$ & Success rate & $t_\mathrm{c}$ (s) \\
\midrule
Full target distribution & 10 & $XX/10$, XX [XX, XX] & XX [XX, XX] \\
\textbf{Ours} & 10 & $XX/10$, XX [XX, XX] & XX [XX, XX] \\
\bottomrule
\end{tabular}
\end{center}
\end{table}
```

Figure: two stills from the video (peg lifting the cover, peg in the hole), side by side like
`fig:door`, label `fig:covered`. Upload them to `img/`.

**Check before submitting**: "about 0.8" comes from the successful trials so far (phase 0.80 to
0.84 at insertion); confirm it on the final set.

## 10. Discussion — limitation 2 DONE 2026-11-13 in the repo manuscript (fifth already in)

Replace the opening sentence's "not conclusive at 48 trials each" only if the new door table
changes it (AFTER DOOR). Replace limitation 2

> Second, the ablations and baselines other than the full target distribution rest on 24
> trials, which leaves intervals of about $\pm20$ points; we rank conditions only where
> intervals do not overlap, and report no significance test.

with

(Superseded 2026-11-13: limitation 2 in the repo manuscript is the +-14-point interval width at
48 trials; the door rig was unchanged.)

Add a fifth limitation:

```latex
Fifth, the covered peg-in-hole rests on 10 trials per condition, which leaves intervals of about
$\pm26$ points, and its success is judged by the operator rather than by the phase.
```

Limitations 1, 3 and 4 stay. Limitation 4 is about the door; leave it as it is.

## 11. Conclusions — NOW, wording AFTER PEG

After "...the ablation identifies the retreat as the component it cannot do without: a target
that only slides forward leaves the arm at a failed subtask." add the covered-peg sentence in the
fixed wording, for example:

```latex
On a second task, a peg-in-hole under a hinged cover, the same method with the same parameters
succeeded in XX of 10 trials.
```

In the last paragraph, replace "Beyond the door, the same formulation applies to any task whose
stages follow one demonstration, such as multi-step assembly." with

```latex
The covered peg-in-hole suggests that the formulation applies to other tasks whose stages follow
one demonstration, such as multi-step assembly.
```

## 12. Appendix placeholder — DONE 2026-11-13: section removed (ACKNOWLEDGMENT still template text)

The `\section*{APPENDIX}` still holds the template text "Appendixes should appear before the
acknowledgment." Either remove the section or use it for the covered peg video description
(one line per clip). Supplementary video is the usual place for it.

## 13. Covered peg numbers — AFTER PEG (needs the repo, not the editing computer)

`final/paper/results_table.py` cannot be reused for the covered peg: `analyse_runs.trials`
counts `abort` as a failure and times a trial by its end. Add a separate function for the
covered peg (abort -> success, time = first time at the trial's maximum phase) instead of
changing `trials`, which the door table depends on. Run files are in
`final/saved_recordings/evaluation_covered_peg_appendix/`. Not counted: `run_e2t2_20261112_102031`
(empty) and `run_e2t2_20261112_103618` (a one-trial run, not a memory sequence). Ask Claude to
do this step in the repo.

## 14. Presentation — NOW (skeleton), numbers later

1. Problem: sequential contact tasks; why plain ergodic control gets stuck.
2. Method: phase labels, sliding target, retreat, visitation decay (reuse
   `final/paper/img/step_*.svg` and `retry_*.svg`).
3. Door: photos, conditions table, success with intervals, survival curve, "the retreat does the
   most work".
4. Covered peg: video (one success per condition, one failure), the two-row table, labelled
   n = 10 and operator-judged.
5. Limitations and next step: the moved lock.

## Order

1. NOW edits: 2, 3, 4, 5, 7, 9 (text), 10, 11 (structure), 12, 14 (skeleton), 1 (except numbers).
2. AFTER PEG: 13, then the XX in 1, 9, 11.
3. AFTER DOOR: checksums of the new runs, `results_table.py`, then 6, 8, the 12 % in 1, and the
   Discussion opening sentence.
4. Last: read every number back to a script output; check names against `GLOSSARY.md`.

## Which conclusions can change at 48 trials (computed 2026-11-13)

Ours (41/48) and the full target distribution (33/48) are final, so "Ours vs full target
distribution: not conclusive" and the abstract and Conclusions cannot change. Ours' lower bound
is 0.73. If a condition's second half repeats its first half:

- Ours w/o Retreat: stays far below whatever happens. "The retreat does the most work" holds.
- Monotonic expansion: stays below Ours unless its second half reaches 17/24.
- Ours w/o Visitation Decay and the single trials: 14/24 again gives [0.44, 0.71], which no
  longer overlaps Ours. Then the Success list moves them and the decay sentence takes the
  `%` alternative ("Both components contribute"). They stay "overlapping" from 15/24 up.

## Open

- Deadline: report handed in Thursday 2026-11-19. Proposed, not yet confirmed: door sessions not done by
  Wednesday 2026-11-18 12:00 are dropped and the report gives the count reached (decided
  before the runs, so it is not a stopping rule on the results). If the door top-up cannot finish before them,
  report the topped-up conditions at the count reached and adjust 5, 6 and 7 to match.
