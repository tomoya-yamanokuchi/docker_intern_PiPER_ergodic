# Phase-augmented ergodic control on the PiPER: implementation summary

Branch `explore/master-backwards-search`.

## Note: this is a deliberate simplification

**The phase is estimated from the TCP position alone, in plain Euclidean metres. That is too simple for the
6-DoF task.** It ignores orientation entirely, so a phase in which the pose only rotates is invisible to the
phase clock. The turn after insertion is such a phase: the position barely changes while the key or peg is
turned. It was chosen because both normalised 6-D metrics stalled the clock on the arm:

- **Dividing each axis by the datapoints' spread** let the wrist's tilt outweigh progress. The arm tilted 25–33°
  (median) against a datapoint spread of about 3°.
- **The same with position only** then let a few millimetres of height outweigh centimetres of progress. The
  datapoints vary 3.6 mm in z against 44–69 mm in x and y.

**What it shows:** with a phase clock that actually follows the arm, the method may work on the real arm. Next we
continue to 6-DoF, where orientation has to enter the phase estimate again, in a metric that neither the tilt
nor a near-constant axis can dominate.

## Goal

The goal is to run the method of the 2-D notebook
(`ergodic_controller/Ergodic_Exploration_phase_key_in_lock_2D.ipynb`) on the arm, with recorded data:

- a master trajectory gives the phase φ;
- the taught datapoints are labelled with φ;
- the E2T2 law is fed a phase-conditioned target and spatial statistic;
- a stall widens the backward kernel σ_b.

## Pipeline

1. **Master:** `record_joint_angles.py`, unchanged. One continuous demonstration of the whole task. φ = t / T after
   the dwell trim of `load_recording`.
2. **Labels:** `label_datapoint_phases.py <master> <datapoints>` writes `<datapoints>_phase.npz` (`q`, `phi`) beside
   the input.
   - Each datapoint gets the φ of its nearest master sample, by `peg_tcp` position in metres.
   - h = the median k-th neighbour distance in metres (k = round(√M)), dℓ/dφ = the master's path length, and
     σ_f = lead = h / (dℓ/dφ).
3. **View:** `visualization/visualize_phase_distribution.py <master> <labelled>`.
   - A MeshCat φ slider (**Animations > default > time**) shows, per phase, the weighted datapoints and the target
     density the law would track: the K = 10 Fourier series of the weighted datapoints, position marginal.
4. **Run on the arm:** `run_ergodic_phase.py <master> <labelled>`. It uses `ergodic_controller/phase_ergodic_controller.py`
   in the unchanged loop of `execution/live_ergodic_controller.py`, with the same gains, rates, MAX_SPEED, torque
   check and exit hold. Every 20th step it prints φ, σ_b and stall/T. LiveView draws the live target density.
   On exit it writes the run and a phase trace.

## The phase-conditioned law

- **Target density:** the datapoints weighted by an asymmetric kernel around φ + lead (σ_f ahead, σ_b behind),
  normalised every step.
- **Spatial statistic:** the arm's past states weighted by phase similarity, with mass 1, as E2T2's time average.
- **Phase clock:** monotone, windowed (φ − 0.05 … φ + 0.12), with σ_f / 4 hysteresis. σ_b = min(σ_f(1 + stall/T), 0.5).
- **No offset carry-forward and no step-back**, as decided in the notebook.
- **Exact evaluation:** the law is computed exactly as a sum over points through Λ's TT cores. There is no K⁶ tensor and
  no per-step TT-cross. With the phase frozen it takes the same steps as `ErgodicController` to 1e-12, with that
  controller's statistic rounding switched off.

## Verification (offline)

- **Tests:** `test_phase_projection.py` (7) and `test_phase_ergodic_controller.py` (5) pass. The full offline suite is
  56/56 and lint is clean.
- **Dry runs:** `simulation/fake_executor_helpers.py` runs `run_ergodic_phase.py` end to end with no loop overruns.
  The φ-slider and LiveView clouds have not been checked in a browser.
- **Replays of recorded runs:** these showed why each metric stalled. Measured in metres, the arm's own progress in
  run `161725` stayed below φ ≈ 0.17, and the new clock follows it.

## Open points

- **Orientation in the phase estimate:** the 6-DoF step (note above).
- **Memory growth during long stalls:** the law's cost grows with the past states in the current phase. A step takes
  4.6 ms at 2000 states and 11 ms at 5000, about 4 minutes stalled in one phase.
- **Joint 6** must be within ±120° before a run: it once read about 1092° and the arm could not be connected.
