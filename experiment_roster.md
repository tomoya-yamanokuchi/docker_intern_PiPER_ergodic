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
