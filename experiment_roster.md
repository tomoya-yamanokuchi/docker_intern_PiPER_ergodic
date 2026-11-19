# Experiment roster: Week 9 comparison and ablation

Design and decisions are in `week9.md`. This file is the procedure and the schedule. Claude does not run any of it; you run it on the arm.

## Before the first session

1. Code is the tag `door-experiment-v4.3`. Check from the repo root: `git describe --tags --exact-match` prints it and `git status --short final/src` shows no changed tracked file. Do not edit `final/src` during the experiment; a change means a new tag and a note here. All six sessions ran on `door-experiment-v4.3`. After the last session (11-06), `door-experiment-v4.4` added only analysis: Fisher's exact test in `analyse_runs.py` and the paper's table script `final/paper/results_table.py`; nothing that runs the arm changed. The run files are listed with SHA-256 in `final/saved_recordings/door/week9_runs.sha256` (paths relative to `final/output`, check with `sha256sum -c` from there); `run_retry_20261102_113800` (an aborted start, no trials) was moved to `evaluation_excluded/` on 11-06.
2. CAN is up (`CLAUDE.md` section 6): `ip -details link show can0` shows UP, bitrate 1000000, and `rx_packets` increases.
3. The arm is powered, E-stop released, not in teaching mode.
4. Door closed, key out, the arm at the master's start. A run that starts away from it needs you: the arm is backdrivable, put it near the start and press Enter (the script prints the distance).
5. MeshCat is optional; the loop does not need it.

`e2t2` runs with its memory: the statistic is handed on across the 4 trials of a sequence, as its paper proposes. "E2T2 w/o memory" is a sixth condition, run as 24 independent one-trial runs of `e2t2` (`... e2t2 <task> 1 --auto`). Each is a new process with an empty statistic, so there is no memory by construction. Its files go to their own directory with `--out=../output/evaluation_nomemory`, because `analyse_runs.py` groups by method name and would otherwise pool them with the 4-trial `e2t2` sequences.

## Running one sequence

This arm reports S-V1.8-6: every command below needs `--fw V183`, including the one-trial `e2t2` runs.

Inside the container, from `final/src`:

```bash
cd /home/jens/workspace/docker_intern_PiPER_ergodic/final/src
sh run.sh run_trials.py <method> ../saved_recordings/door/task_20261023_131342.npz 4 --auto --fw V183
```

`<method>` is one of `e2t2`, `expanding`, `sliding`, `retry`, `retry_nodecay`. The 4 is the sequence length.

- `--auto`: a 2 s hold at the master's start, then the trial starts. After every trial the master is played back to close the door and take the key out, so no hand is needed unless a reset fails.
- A trial ends itself at phase 0.9 (success) or 600 s (limit). Enter during a trial aborts it; use it only if you had to touch the arm.
- Ctrl-C holds the arm and saves the trials so far, and the script exits with 130, which ends a shell loop that uses `|| break`.
- One file per sequence: `final/output/evaluation/run_<method>_<timestamp>.npz`.

Chain several sequences with a loop, one method per call:

```bash
for m in expanding retry e2t2 sliding retry_nodecay; do
  sh run.sh run_trials.py $m ../saved_recordings/door/task_20261023_131342.npz 4 --auto --fw V183 || break
done
```

The statistic is handed on across the 4 trials only for `e2t2` and `expanding`; the others start each trial fresh.

## Roster

Sample unit: the sequence of 4 trials. Targets: 12 sequences (48 trials) for `e2t2` and `retry`; 6 sequences (24 trials) each for `expanding`, `sliding` and `retry_nodecay`. Fixed n, no early stop.

Already on disk (10-29, tags v4 and v4.1, rig-only differences, pooled): 5 full `e2t2` sequences and 4 full `retry` sequences of 4 trials each. The 10-30 files include 13 one-trial `e2t2` runs (timestamps 124430, 130714, 131215, 132142, 133157, 135826, 140317, 141336, 144002, 155813, 161106, 163150, 164826), run on the same code as v4.1 (no commits on 10-30). They are counted as no-memory runs: move them to `final/output/evaluation_nomemory/` before analysis so they leave the 4-trial `e2t2` pool. The other 10-30 files (interrupted runs, one abort, one-trial `retry` runs) are excluded. Confirmed by the user: the 13 count. Moved on 11-02: the 13 to `final/output/evaluation_nomemory/`, the other 15 10-30 files to `final/output/evaluation_excluded/`, so `evaluation/` holds only the 9 full sequences and `run_*.npz` there is safe to analyse.

| Condition | Target | Done | Remaining |
|---|---|---|---|
| `e2t2` | 12 | 12 | 0 |
| `retry` | 12 | 12 | 0 |
| `expanding` | 6 | 6 | 0 |
| `sliding` | 6 | 6 | 0 |
| `retry_nodecay` | 6 | 6 | 0 |
| `e2t2` w/o memory (1-trial runs) | 24 | 24 | 0 |

33 sequences, 132 trials, plus 11 one-trial `e2t2` runs. At about 3 min per successful retry trial and up to 10 min per limit trial plus a reset, budget roughly 1.5 to 2.5 h per session.

Six sessions spread the conditions over time, so drift (a looser fixture, a warmer arm) hits all conditions alike. Every session contains `expanding`, `sliding` and `retry_nodecay` once. The order rotates, so no condition is always first or last.

| Session | Sequence order (method x count) |
|---|---|
| 1 | `retry` x2, `e2t2` x2, `expanding`, `sliding`, `retry_nodecay` |
| 2 | `sliding`, `e2t2`, `retry_nodecay`, `retry` x2, `expanding` |
| 3 | `retry_nodecay`, `retry`, `expanding`, `e2t2`, `sliding` |
| 4 | `e2t2`, `expanding`, `retry`, `sliding`, `retry_nodecay` |
| 5 | `expanding`, `retry`, `sliding`, `e2t2`, `retry_nodecay` |
| 6 | `retry`, `retry_nodecay`, `sliding`, `e2t2`, `expanding` |

Add 2 one-trial `e2t2` runs (`e2t2 ... 1 --auto --fw V183 --out=../output/evaluation_nomemory`) to each of sessions 1 to 5 and 1 to session 6 (11 in all), as the last block of the session. Totals: `e2t2` 2+1+1+1+1+1 = 7, `retry` 2+2+1+1+1+1 = 8, the three others 6 each. Tick a row when its session is done and note the file names.

| Session | Date | Files (timestamps) | Notes |
|---|---|---|---|
| 1 | 2026-11-02 | retry x2: `evaluation_retry/run_retry_20261102_134658`; e2t2 x2: `evaluation_e2t2/run_e2t2_20261102_141449_t1-4`, `run_e2t2_20261102_143200`; expanding: `evaluation_expanding/run_expanding_20261102_152106` | retry x2 run as one 8-trial file (retry starts each trial fresh, so it counts as 2 sequences); outcomes limit + 7 success; all resets reached the start. e2t2 was run as one 8-trial sequence (memory spans 8, so only trials 1-4 count): trials 1-4 split into `_t1-4` (4 success), the original 8-trial file moved to `evaluation_excluded/` (trials 5-8: 3 success, 1 limit). Second e2t2 rerun as 4 trials (143200: 4 success, resets ok). expanding: limit, success, limit, limit; resets ok. sliding: 4 limits; resets ok. retry_nodecay: success, limit, success, success; resets ok. e2t2 1-trial x2: success, success; resets ok. Done. Per-method folders `evaluation_<method>/` used instead of `evaluation/`. |
| 2 | 2026-11-02, 11-04 | sliding: `evaluation_sliding/run_sliding_20261102_201246`; e2t2: `evaluation_e2t2/run_e2t2_20261102_204343`; retry_nodecay: `evaluation_retry_nodecay/run_retry_nodecay_20261104_100729`; retry x2: `evaluation_retry/run_retry_20261104_103749`, `run_retry_20261104_104939`; expanding: `evaluation_expanding/run_expanding_20261104_112608`; e2t2 1-trial x2: `evaluation_nomemory/run_e2t2_20261104_113759`, `run_e2t2_20261104_114417` | sliding: limit, success, limit, limit; resets ok. e2t2: limit, success, success, success; resets ok. retry_nodecay (11-04, session continued on a new day): success, success, success, limit; resets ok. retry (1 of 2): limit, success, success, success; resets ok. retry (2 of 2): 4 success; resets ok. expanding: success, limit, success, success; resets ok. e2t2 1-trial x2: success, success; resets ok. Done. |
| 3 | 2026-11-04 | retry_nodecay: `evaluation_retry_nodecay/run_retry_nodecay_20261104_121009`; retry: `evaluation_retry/run_retry_20261104_131705`; expanding: `evaluation_expanding/run_expanding_20261104_140034`; e2t2: `evaluation_e2t2/run_e2t2_20261104_142641`; sliding: `evaluation_sliding/run_sliding_20261104_145450`; e2t2 1-trial x2: `evaluation_nomemory/run_e2t2_20261104_153338`, `run_e2t2_20261104_160033` | retry_nodecay: success, success, success, limit; resets ok. retry: 4 success; resets ok. expanding: 4 limits (trial 1 reached phase 0.84); resets ok. e2t2: limit, success, success, success; resets ok. sliding: success, limit, success, limit; resets ok. e2t2 1-trial x2: success, limit; resets ok. Done. |
| 4 | 2026-11-04, 11-05 | e2t2: `evaluation_e2t2/run_e2t2_20261104_164534`; expanding: `evaluation_expanding/run_expanding_20261105_103958`; retry: `evaluation_retry/run_retry_20261105_112218`; sliding: `evaluation_sliding/run_sliding_20261105_131047`; retry_nodecay: `evaluation_retry_nodecay/run_retry_nodecay_20261105_134705`; e2t2 1-trial x2: `evaluation_nomemory/run_e2t2_20261105_135507`, `run_e2t2_20261105_140212` | e2t2: limit, success, success, success. The reset after trial 1 failed (playback hit the 60 s reset limit); a 1.5 s walk then reached the start (presumably after a reset by hand; not confirmed). expanding (11-05): limit, success, limit, success; again the reset after trial 1 hit the 60 s limit, then a 4.5 s walk reached the start. retry: limit, success, success, limit; resets ok. sliding: 4 limits; the resets after trials 1 and 3 hit the 60 s limit. retry_nodecay: limit, limit, limit, success; resets ok (those after trials 1 and 3 took only 0.5 and 0.8 s). e2t2 1-trial x2: success, success; resets ok. Done. Session weaker across all methods than 1-3, with 4 failed resets: suspect fixture drift. |
| 5 | 2026-11-05 | expanding: `evaluation_expanding/run_expanding_20261105_142819`; retry: `evaluation_retry/run_retry_20261105_145025`; sliding: `evaluation_sliding/run_sliding_20261105_155450`; e2t2: `evaluation_e2t2/run_e2t2_20261105_183357`; retry_nodecay: `evaluation_retry_nodecay/run_retry_nodecay_20261105_185721`; e2t2 1-trial x2: `evaluation_nomemory/run_e2t2_20261105_191236`, `run_e2t2_20261105_191509` | expanding: success, limit, success, success; resets ok. retry: limit, success, success, success; resets ok. sliding: 4 limits; the resets after trials 1 and 3 hit the 60 s limit (same as session 4's sliding). e2t2 (18:33, after a break): 4 limits, the first all-limit e2t2 sequence; resets ok, but the first walk to the start took 4.2 s (usually 0.5 s). retry_nodecay: limit, success, success, success; resets ok (the one after trial 1 took 0.5 s). e2t2 1-trial x2: limit, success; resets ok. Done. |
| 6 | 2026-11-06 | retry: `evaluation_retry/run_retry_20261106_094345`; retry_nodecay: `evaluation_retry_nodecay/run_retry_nodecay_20261106_102433`; sliding: `evaluation_sliding/run_sliding_20261106_132642`; e2t2: `evaluation_e2t2/run_e2t2_20261106_141155`; expanding: `evaluation_expanding/run_expanding_20261106_144502`; e2t2 1-trial: `evaluation_nomemory/run_e2t2_20261106_145522` | retry: success, success, success, limit (successes slow: 584, 540, 231 s); resets ok. retry_nodecay: limit, limit, success (520 s), limit; resets ok. sliding: 4 limits; the reset after trial 4 hit the 60 s limit. e2t2: 4 limits, first walk 4.3 s (like session 5's all-limit e2t2, 4.2 s; every other e2t2 sequence since 11-02 started at 0.5-0.6 s). expanding: success, success, success, limit; the reset after trial 4 hit the 60 s limit. e2t2 1-trial: success (150 s), first walk 5.0 s; resets ok. Done. Roster complete. |

## Top-up to 48 trials per condition (sessions 7 to 12)

Decided 2026-11-12: every condition goes to 48 trials, the count `e2t2` and `retry` already have. 48 rather than 50 because `e2t2` and `expanding` carry their statistic across a 4-trial sequence, so 50 would leave a 2-trial remnant.

| Condition | Target | Done | Remaining |
|---|---|---|---|
| `e2t2` | 12 seq | 12 | 0 |
| `retry` | 12 seq | 12 | 0 |
| `expanding` | 12 seq | 12 | 0 |
| `sliding` | 12 seq | 12 | 0 |
| `retry_nodecay` | 12 seq | 12 | 0 |
| `e2t2` w/o memory (1-trial runs) | 48 | 48 | 0 |

18 sequences (72 trials) plus 24 one-trial runs, roughly 2 h per session, so 12 h in all.

**Code.** Sessions 1 to 6 ran on `door-experiment-v4.3`. `final` has since moved on: `c16a5ea` (closing trajectory, a no-op for the door) and an uncommitted `run_trials.py` change that stretches the walk limit with the walk's length (`RESET_SLACK`), which changes the door's reset timeout from 60 s to about 64 s. So the top-up runs from a separate checkout of `door-experiment-v4.4`, whose arm code is identical to v4.3, at `.claude/worktrees/door-v4.4` (detached, made 2026-11-12). The `final` working tree, with the covered-peg changes, is not touched. `--out` takes an absolute path, so the runs land in the same per-method folders as sessions 1 to 6.

**Rig.** The door fixture was taken down for the covered peg-in-hole, and these runs happen after it is rebuilt. The master and the task file assume the lock where it was taught on 10-23. Before session 7, run one `retry` sequence into `evaluation_rigcheck/`. It does not count toward any condition, and that is decided before seeing it. If it is much worse than retry's 41/48 (say, 2 or more limits, or a failed reset), check the fixture position before going on. All six conditions then have runs from before and after the rebuild, but only three of them got trials after it. Note this as a limitation, and do not claim a rig effect either way.

Check the checkout on the host (git in the container cannot follow the worktree's host path): `git -C .claude/worktrees/door-v4.4 describe --tags --exact-match` prints `door-experiment-v4.4` and `git -C .claude/worktrees/door-v4.4 status --short` prints nothing. Then, inside the container:

```bash
cd /home/jens/workspace/docker_intern_PiPER_ergodic/.claude/worktrees/door-v4.4/final/src
TASK=../saved_recordings/door/task_20261023_131342.npz
OUT=/home/jens/workspace/docker_intern_PiPER_ergodic/final/output

# rig check, before session 7 only
sh run.sh run_trials.py retry $TASK 4 --auto --fw V183 --out=$OUT/evaluation_rigcheck

# one session: replace the three methods with that session's order from the table below
for m in expanding sliding retry_nodecay; do
  sh run.sh run_trials.py $m $TASK 4 --auto --fw V183 --out=$OUT/evaluation_$m || break
done
for i in 1 2 3 4; do
  sh run.sh run_trials.py e2t2 $TASK 1 --auto --fw V183 --out=$OUT/evaluation_nomemory || break
done
```

Each session has one sequence of each of the three conditions, in a rotated order (each condition goes first, middle and last twice). The four one-trial runs are the last block, as in sessions 1 to 6. A session may be split across a break. Make the split at a sequence boundary, never inside one.

| Session | Order | Then | Date | Files (timestamps) | Notes |
|---|---|---|---|---|---|
| 7 | `expanding`, `sliding`, `retry_nodecay` | 4 x 1-trial | 2026-11-13 | expanding: `evaluation_expanding/run_expanding_20261113_142626`; sliding: `evaluation_sliding/run_sliding_20261113_145841`; retry_nodecay: `evaluation_retry_nodecay/run_retry_nodecay_20261113_152150`; e2t2 1-trial x4: `evaluation_nomemory/run_e2t2_20261113_153215`, `153825`, `154221`, `155359` | Rig check (not counted), `evaluation_rigcheck/`: `run_retry_20261113_132046` interrupted (621 KB, no trials); `run_retry_20261113_133318` 4 success, so the rig passes. Also in `evaluation_rigcheck/`: `run_e2t2_20261113_140751` (limit, limit, limit, success), an e2t2 sequence not in the plan, not counted. expanding: 4 success. sliding: limit, limit, limit, success. retry_nodecay: success, success, limit, success. e2t2 1-trial x4: limit, success, success, limit; the reset after the last hit the 60 s limit (the others took 0.5-0.6 s to the start). Done. |
| 8 | `sliding`, `retry_nodecay`, `expanding` | 4 x 1-trial | 2026-11-16 | sliding: `evaluation_sliding/run_sliding_20261116_104133`; retry_nodecay: `evaluation_retry_nodecay/run_retry_nodecay_20261116_110836`; expanding: `evaluation_expanding/run_expanding_20261116_114250`; e2t2 1-trial x4: `evaluation_nomemory/run_e2t2_20261116_122906`, `123125`, `124147`, `125320` | sliding: limit, limit, limit, success; one reset hit the 60 s limit. retry_nodecay: 4 success; resets ok (13.1-13.2 s). expanding: success, limit, limit, limit; one reset hit the 60 s limit. e2t2 1-trial x4: success, success, limit, limit; resets ok. Done. |
| 9 | `retry_nodecay`, `expanding`, `sliding` | 4 x 1-trial | 2026-11-16 | retry_nodecay: `evaluation_retry_nodecay/run_retry_nodecay_20261116_131053`; expanding: `evaluation_expanding/run_expanding_20261116_135714`; sliding: `evaluation_sliding/run_sliding_20261116_143817`; e2t2 1-trial x4: `evaluation_nomemory/run_e2t2_20261116_144237`, `144534`, `144903`, `145851` | retry_nodecay: success, limit, success, success; resets ok. expanding: 4 limits; the reset after trial 1 hit the 60 s limit without reaching the start. sliding: limit, success, limit, limit; the reset after trial 3 hit the 60 s limit without reaching the start. e2t2 1-trial x4: 4 success; resets ok. Done. |
| 10 | `expanding`, `retry_nodecay`, `sliding` | 4 x 1-trial | 2026-11-16 | expanding: `evaluation_expanding/run_expanding_20261116_153046`; retry_nodecay: `evaluation_retry_nodecay/run_retry_nodecay_20261116_160229`; sliding: `evaluation_sliding/run_sliding_20261116_165308`; e2t2 1-trial x4: `evaluation_nomemory/run_e2t2_20261116_170330`, `171629`, `171826`, `172850` | expanding: success, success, limit, limit; resets ok. retry_nodecay: success, success, limit, success; resets ok. sliding: 4 limits; the reset after trial 2 hit the 60 s limit without reaching the start. e2t2 1-trial x4: limit, limit, success, limit; resets ok. Done. |
| 11 | `sliding`, `expanding`, `retry_nodecay` | 4 x 1-trial | 2026-11-16, 11-17 | sliding: `evaluation_sliding/run_sliding_20261116_181042`; expanding: `evaluation_expanding/run_expanding_20261116_184917`; retry_nodecay: `evaluation_retry_nodecay/run_retry_nodecay_20261116_195710`; e2t2 1-trial x4 (11-17, session continued on a new day): `evaluation_nomemory/run_e2t2_20261117_101405`, `101942`, `102632`, `102919` | sliding: 4 limits; resets ok. expanding: limit, success, success, success; resets ok. retry_nodecay: success, limit, success, success; resets ok. e2t2 1-trial x4: limit, success, success, success; resets ok. Done. |
| 12 | `retry_nodecay`, `sliding`, `expanding` | 4 x 1-trial | 2026-11-17 | retry_nodecay: `evaluation_retry_nodecay/run_retry_nodecay_20261117_105605`; sliding: `evaluation_sliding/run_sliding_20261117_113128`; expanding: `evaluation_expanding/run_expanding_20261117_124517`; e2t2 1-trial x4: `evaluation_nomemory/run_e2t2_20261117_124934`, `125418`, `130727`, `131028` | retry_nodecay: success, success, success, limit; resets ok. sliding: success, limit, success, limit; resets ok. expanding: success, limit, success, success; resets ok. e2t2 1-trial x4: success, success, limit, success; resets ok. Done. Top-up complete; checksums of sessions 7-12 appended to `week9_runs.sha256`. |

After session 12, append the new files' SHA-256 to `final/saved_recordings/door/week9_runs.sha256` (or a new `.sha256` beside it), and rerun `final/paper/results_table.py`, whose numbers in the paper then change.

## Between sequences

Nothing to do if the reset worked (it prints whether it reached the start). If it failed, take the key out by hand, close the door, put the arm near the master's start and press Enter when prompted. Note it in the session table.

If a session dies halfway (CAN, E-stop, power), the sequence in progress is saved as far as it went. A sequence with fewer than 4 trials is not a full sequence: rerun it from the start and note the partial file as excluded.

## Analysis

From the repo root, after each session:

```bash
docker exec piper-ergodic bash -lc 'cd /home/jens/workspace/docker_intern_PiPER_ergodic/final/src && sh run.sh analyse_runs.py ../output/evaluation/run_*.npz ../output/evaluation_{e2t2,retry,expanding,sliding,retry_nodecay}/run_*.npz'
docker exec piper-ergodic bash -lc 'cd /home/jens/workspace/docker_intern_PiPER_ergodic/final/src && sh run.sh analyse_runs.py ../output/evaluation_nomemory/run_*.npz'
```

Since session 1 every condition has its own folder, `final/output/evaluation_<method>/`; `evaluation/` holds only the 9 sequences from 10-29. The first command pools them per method; the second is "E2T2 w/o memory", kept apart because its files also carry the method `e2t2`. Never put `evaluation_excluded/` in either. It groups by the method stored in each file and prints the six metrics per method with Wilson 95 % intervals, then writes `survival.png`. Do not select files with `run_retry_*`: it also matches `run_retry_nodecay_*`. Use `run_*.npz`, which groups correctly, or `run_retry_2*.npz` for plain retry.

Ranking for the ablation: success-rate drop from `retry` to `sliding` (what the retreat adds) and from `retry` to `retry_nodecay` (what the forgetting adds), against `expanding` and `e2t2` as baselines. The analysis has no between-method test yet; add one before the final paper if intervals overlap.

## Open

- `retry_nodecay` and `expanding` have never run on the arm, only the existing offline tests. Run `sh run.sh simulate.py retry_nodecay 1` and `... expanding 1` (fake arm, safe) before session 1.
- `run_trials.py`'s usage line does not list `retry_nodecay`; it is accepted anyway because the check reads `METHODS`.

## Covered peg-in-hole (internship report, 2026-11-12)

A second task for the report, not part of the door experiment. Arm S-V1.8-2 (no `--fw`), run from the main `final/src` (it needs the closing trajectory, `c16a5ea`), task `task_20261111_194118`. Every trial is filmed.

**Conditions.** `e2t2` with memory: 2 sequences of 5 trials, the statistic handed on within a sequence. This is the strongest baseline from the door. `retry` (Ours): 10 trials. It starts every trial fresh, so how they are batched does not matter.

**Success rule, fixed before the remaining trials.** The operator presses Enter once the peg is in the hole. The file records that as `abort`, and in this folder every `abort` is a success, so press Enter for nothing else and use Ctrl-C to stop for any other reason. A trial that hits the 600 s limit is a failure. Success time is when the phase first reaches its trial maximum. Two corrections, `run_retry_20261112_105210` and `run_retry_20261112_161814`: Enter was missed on an insertion confirmed on video, so their outcomes were set to `success` on 11-12, each with a `note` in the file and the original kept as `.npz.orig`.

**On disk** (`final/saved_recordings/evaluation_covered_peg_appendix/`): `retry` 10 of 10, done: 6 successes (105210 402.5 s, 105711 165.3 s, 110602 364.0 s, 155103 358.7 s, 161814 455.3 s, 20261113_105117 131.0 s), 4 limits (154327, 160223, 163120, 20261113_110522). `e2t2` 10 of 10, done: 0 successes. Moved to `excluded/` on 11-12: `run_e2t2_20261112_102031` (empty start), `run_e2t2_20261112_103618` (a one-trial run, not a memory sequence), `run_e2t2_20261112_123318` (interrupted at 74.6 s).

**Order** (alternating, so drift over the day hits both alike):

| Block | Run | Date | File | Outcomes |
|---|---|---|---|---|
| 1 | `e2t2` sequence of 5 | 2026-11-12 | `run_e2t2_20261112_135032` | 5 limits (highest phase 0.44-0.52); resets ok |
| 2 | `retry` x4 | | | |
| 3 | `e2t2` sequence of 5 | 2026-11-12 | `run_e2t2_20261112_144825` | 5 limits (highest phase 0.23-0.52); resets ok |
| 4 | `retry` x3 | | | |

Inside the container. `$CAN` is the interface the S-V1.8-2 arm is on: `can0` with one adapter plugged in, otherwise check it with the `rx_packets` test.

```bash
cd /home/jens/workspace/docker_intern_PiPER_ergodic/final/src
TASK=../saved_recordings/covered-peg-in-hole/task_20261111_194118.npz
OUTC=../saved_recordings/evaluation_covered_peg_appendix
sh run.sh run_trials.py e2t2  $TASK 5 --auto --can $CAN --out=$OUTC   # blocks 1 and 3
sh run.sh run_trials.py retry $TASK 4 --auto --can $CAN --out=$OUTC   # block 2; 3 for block 4
```

A sequence cut short (Ctrl-C, CAN, E-stop) is not a full sequence: rerun it from the start and move the partial file to `excluded/`. About 1.5 to 2 h for the e2t2 sequences (trials that reach the limit take 10 min) plus about 50 min of retry.

### Covered peg-in-hole ablation (2026-11-17)

The same task and arm as above, now with the door's four other conditions so the covered peg has all six. `retry` and `e2t2` (10 trials each) are on disk and are not rerun. Two sessions of 5 trials per condition, 10 in all.

**The success rule differs from the appendix runs.** A trial ends itself as a success once the phase passes 0.8 (`--success-phase=0.8`), so Enter is no longer used for success. After the sessions, check each `success` on the video; a trial that was not an insertion is changed to a failure (a `note` in the file, the original kept as `.npz.orig`, as before). Enter and Ctrl-C still stop a trial and are not successes. The reset after a success is the door's (the offset fades along the master); the appendix runs' Enter-aborts were reset without the fade.

Each condition has its own folder, so `analyse_runs.py` does not pool them. `expanding` hands its statistic on, so a session's 5 trials are one sequence (session 2 starts a new one). The no-memory runs are 1-trial `e2t2` runs, as on the door.

```bash
cd /home/jens/workspace/docker_intern_PiPER_ergodic/final/src
TASK=../saved_recordings/covered-peg-in-hole/task_20261111_194118.npz
OUT=../saved_recordings/evaluation_covered_peg_ablation
F="--auto --can $CAN --success-phase=0.8"

# one session; session 2 runs the first loop in the order retry_nodecay expanding sliding
for m in sliding retry_nodecay expanding; do
  sh run.sh run_trials.py $m $TASK 5 $F --out=$OUT/evaluation_$m || break
done
for i in 1 2 3 4 5; do
  sh run.sh run_trials.py e2t2 $TASK 1 $F --out=$OUT/evaluation_nomemory || break
done
```

A cut-short `expanding` sequence is rerun from the start, with the partial file moved to `excluded/`. A limit trial takes 10 min, so budget 3 to 4 h per session if most of them limit.

| Session | Date | Files | Notes |
|---|---|---|---|
| 1, with damping | 2026-11-17 | all moved to `evaluation_covered_peg_ablation/with_damping/` | Run with the damping addition, which was then removed, so these do not count. 5 per condition: sliding 5/5 (20-23 s; one Enter abort removed), retry_nodecay 0/5, expanding 5/5, e2t2 w/o memory 5/5 (28-103 s; two 0-1 s aborts excluded). Kept as a record; `sliding` trial 2 removal is in its `.npz.orig`. |
| 1, redo | | | |
| 2 | | | |

**Session 1 redo, without the damping addition.** The full block again: 5 `sliding`, 5 `retry_nodecay`, 5 `expanding`, 5 one-trial `e2t2`. Same `TASK`, `OUT`, `F` and the first-session loop above, which writes to fresh `evaluation_<method>/` folders now that the old ones are in `with_damping/`. `retry` and `e2t2` with memory (11-12/13) were run with the damping addition or not, and must be re-checked for comparability before they sit in the same table.

Enter only for a trial you had to touch, and not at the start. A trial that aborts early is excluded, not counted as a failure, and rerun.

**Open before session 2.** The video review of each `success`, and whether the fixture is where it was taught. Session 2 is then the same block in the order `retry_nodecay expanding sliding`, then 5 no-memory runs.

**Totals when done** (valid trials): `sliding` 10, `retry_nodecay` 10, `expanding` 10 (2 sequences of 5), `e2t2` w/o memory 10.
