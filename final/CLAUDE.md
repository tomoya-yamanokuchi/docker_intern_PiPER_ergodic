# final/ — the IROS implementation

A tree of its own: `final/src` never imports `workspace/src`, and runs only through
`final/src/run.sh`, which puts `final/src` and `final/src/agx_reference` on the path
(`robot.py` refuses the sandbox's `core`). Vocabulary is `GLOSSARY.md`; use its terms.

## Where things are

- **The method**: `src/method/law.py` (`Law`; `METHODS` names every method a run takes),
  `task.py` (labelling, `Task`), `pose.py` (the pose metric).
- **Its source**: `src/notebooks/Sequential_E2T2_peg_slide_2D.ipynb`. `law.py` is a port
  of it, and `src/method/test_against_notebook.py` holds the two together. The experiments
  notebook writes its results to `final/output/`.
- **The arm**: `src/arm/piper.py` is the only code that talks to the SDK; `arm/control.py`
  (rates, gains, torque laws) and every script importing either can move the arm.
  `arm/fake.py` stands in for it offline.
- **Scripts**, in pipeline order: `record_master.py` (ARM), `teach_exploration.py` (ARM),
  `label.py`, `run_trials.py` (ARM), `view_master.py` / `view_labels.py` / `view_run.py`.
  `simulate.py` runs the whole chain on the fake arm. Each docstring gives its usage.
- **Kept data**: `saved_recordings/door/` holds the master, exploration and task of the
  first unassisted success (tag `door-working-v1`); `final/output/` is gitignored.
- **Offline analysis of an arm run**: `final/output/run_<method>_<ts>.npz`. Read it with the
  hardware-free modules: `robot.measured_state` turns a logged `q` into the law's state.

## Checks

`final/check.sh` runs ruff and every `test_*.py` in the container; the pre-commit hook runs
it when `final/` is staged. A PreToolUse hook (`.claude/hooks/arm-guard.py`) refuses a Bash
command that runs Python naming an arm module, so keep offline scripts on the modules above
and split a read of an ARM file from a Python run.

## Known differences between the notebook and `law.py`

Deliberate, so a result from one does not carry over unchecked:

- **Statistic across trials**: the notebook hands it on through a sequence for every
  method; `Law.new_trial` starts a fresh one for every forgetting method, only E2T2's is
  handed on.
- **Retry's retreat rate**: the notebook retreats `delta / T_f` phase per second
  (GLOSSARY "Retry" describes this); `law.py` retreats `MAX_SPEED / S_m`, the arm's own
  speed along the master.
- **The tube**: `law.py`'s phase holds while the pose is outside it (GLOSSARY "Tube");
  the notebook's filter has none, and `test_against_notebook` compares the filter without it.
- **The lead and the cube clamp** exist only in `law.py` (the arm lags; the notebook's
  world does not).
- **6-D numerics** (gradient contraction, Lambda unrounded, statistic rounding): the
  `law.py` module docstring lists them.
