# `two_gaps_obstructed`: a sequential planar task solved by phase-conditioned ergodic control

Branch `explore/master-backwards-search`, state at `1fa7a3d` (2026-10-06).

This document explains the pipeline and the algorithm behind the first successful runs of the
`two_gaps_obstructed` demo on the PiPER, and assesses where it stands. The general method summary is
`summary.md`. Parts of it are out of date: it describes a 6-D run, but the code now runs a planar law.

## 1. The task

The peg tip (`peg_tcp`) has to travel a route through the environment in a fixed order. It cannot jump
ahead: a later part of the route can only be reached once an earlier part has been passed. One of the
two candidate passages is obstructed, so the controller has to give up on it and use the other one.
Nothing in the controller detects the obstruction.

**The physical layout is not written down anywhere in the repo.** What follows is reconstructed from the
master, the datapoints and the runs, in world coordinates (mm):

| master phase φ | peg tip (x, y) | segment |
|---|---|---|
| 0.0 | (219, −130) | start region |
| 0.0 → 0.3 | → (288, −129) | +x to the right lane |
| 0.3 → 0.7 | → (296, 66) | +y up the right lane, x ≈ 295 |
| 0.7 → 1.0 | → (375, 60) | +x to the goal |

z stays within about ±15 mm for the whole task, so the task is planar in x-y. The datapoints also cover
a second, left lane at x ≈ 200–240 mm. In every successful run the arm went up that left lane to
y ≈ 65 mm and could not continue in +x from there. That is consistent with the left passage being the
obstructed one. The plain `two_gaps` task is the same layout without the obstruction, and its runs went
start to φ ≈ 0.95 in 17 s with no stall.

## 2. Data

All of it is tracked in `workspace/saved_recordings/`:

| file | content |
|---|---|
| `master/two_gaps_obstructed/joint_angles_demo.npz` | the master: one continuous demonstration of the whole route. 990 raw samples, 482 after the dwell trim, T = 4.85 s |
| `exploration/two_gaps_obstructed/datapoints.npz` | 136 Enter-confirmed datapoints, the first exploration set |
| `exploration/two_gaps_obstructed/datapoints_demo.npz` | 168 datapoints: those 136 unchanged, plus 32 more (checked with `array_equal`) |
| `…/datapoints_phase.npz`, `…/datapoints_demo_phase.npz` | the same sets with phase labels φ. The shared 136 have identical labels |

The 32 added datapoints were all labelled φ ∈ [0.60, 1.00], 26 of them at φ ≥ 0.78. They filled the end of
the task, which the first set had left almost empty:

| φ decile | 0 | .1 | .2 | .3 | .4 | .5 | .6 | .7 | .8 | .9 |
|---|---|---|---|---|---|---|---|---|---|---|
| `datapoints` (136) | 46 | 13 | 14 | 10 | 13 | 14 | 4 | 16 | 5 | 1 |
| `datapoints_demo` (168) | 46 | 13 | 14 | 10 | 13 | 15 | 6 | 22 | 22 | 7 |

The successful runs logged their datapoint file as `saved_recordings/…/datapoints_20261006_124858_phase.npz`.
That is `datapoints_demo_phase.npz` under its old name, so the `recording` field stored in those runs no
longer resolves to a file.

## 3. Pipeline

```
record_joint_angles.py         -> master   joint_angles_*.npz    (q(t), one pass through the task)
teach_datapoints.py            -> cloud    datapoints_*.npz      (Enter-confirmed q, no t)
label_datapoint_phases.py      -> labels   datapoints_*_phase.npz (q, phi)
visualize_phase_distribution.py            MeshCat phi slider: the target the law will see per phase
run_ergodic_phase.py           -> run      ergodic_run_*_phase.npz + phase_trace_*.npz
```

The commands, run from `workspace/src`. The first, second and fifth move the arm, so only the user runs them:

```bash
python record_joint_angles.py --out ../saved_recordings/master/two_gaps_obstructed
python teach_datapoints.py --out ../saved_recordings/exploration/two_gaps_obstructed
python label_datapoint_phases.py <master>.npz <datapoints>.npz
python visualization/visualize_phase_distribution.py <master>.npz <datapoints>_phase.npz
python run_ergodic_phase.py <master>.npz <datapoints>_phase.npz
```

### 3.1 Master → phase

`direct_teaching/distribution/phase_projection.py`. The master is trimmed by `load_recording` and its
phase is its normalised time, φ = t / T. Normalised time is used rather than arc length so that a pause
in the demonstration keeps its share of φ.

### 3.2 Datapoints → labels

`label_datapoint_phases.py`. Each datapoint gets the φ of its nearest master sample, measured by `peg_tcp`
position in plain metres. Three quantities are then derived from the data, with no tuning:

- **h**: the median distance to the k-th nearest neighbour among the datapoints, with k = round(√M).
- **dℓ/dφ**: the master's path length.
- **σ_f = h / (dℓ/dφ)**: the kernel width in phase.

For the demo set this gives M = 168, k = 13, h = 20.3 mm, dℓ/dφ = 452 mm, σ_f = 0.045 and
lead = σ_f = 0.045. For the 136-point set, σ_f = lead = 0.050.

### 3.3 Cube and planar restriction

`execution/live_ergodic_controller.prepare_phase_exploration`. A GMM is fitted to the datapoints only to
define the `[0, 1]^6` cube of `PoseDistribution`, so the gains, `MAX_SPEED` and the interpolator see the
same coordinates as in `run_ergodic_pipeline.py`. The law itself uses the datapoints directly.

The run is **planar** (`PhaseTask.axes = 2`):

- The law steps cube x-y only.
- Every setpoint carries the start pose's z and orientation unchanged.
- Rotation about the peg is left free (`free_peg_rotation`). Each cycle, the commanded orientation is the
  measured one, tilted the shortest way onto the commanded peg axis. The impedance law therefore holds
  the tilt and neither holds nor drives the rotation about the peg.
- Joint 6, which turns about the peg axis, gets a soft limit spring in the last 0.3 rad before each
  URDF limit (`joint6_limit_torque`).

## 4. The algorithm: E2T2 fed a phase-conditioned target

`ergodic_controller/phase_ergodic_controller.py`. The control law is E2T2's, unchanged:

$$b_i = \sum_{\mathbf k} \Lambda_{\mathbf k}\,(\mathcal W_{\mathbf k} - \hat{\mathcal W}_{\mathbf k})\,\nabla_i \Phi_{\mathbf k}(x),
\qquad u = -u_{\max}\,\frac{b}{\lVert b\rVert}$$

What changes is what the law is fed. Both sides are recomputed every step from the current phase φ.

**Target density Ŵ(φ).** The datapoints, weighted by an asymmetric kernel centred ahead of the current phase:

$$w_j \propto \exp\!\Big(-\tfrac{d_j^2}{2\sigma^2}\Big),\quad d_j = \varphi_j - \min(1, \varphi + \sigma_f),
\quad \sigma = \begin{cases}\sigma_f & d_j > 0\\ \sigma_b & d_j \le 0\end{cases},\quad \textstyle\sum_j w_j = 1$$

Centring the kernel σ_f ahead of φ (`LEAD_SCALE = 1`) is the forward pull along the task. Datapoints at the
current phase keep e^(−1/2) ≈ 0.61 of the peak weight.

**Spatial statistic W(φ).** The arm's own past states, weighted by how close their phase is to the current one:

$$v_m = \exp\!\Big(-\tfrac{(\varphi_m - \varphi)^2}{2\sigma_f^2}\Big),\quad \mathcal W(\varphi) = \sum_m \tfrac{v_m}{\sum v}\,\Phi(x_m)$$

Memory from other phases fades out without an explicit decay. With φ frozen, W is exactly E2T2's time
average.

**Phase clock.** `_advance_phase` updates φ each step:

- It projects the measured x-y onto the master, by nearest sample in metres.
- It only searches a window, φ − 0.05 … φ + 0.12.
- φ only ever moves forward, and only by more than σ_f / 4 (hysteresis).
- φ is never a decision variable of the law.

**Stall → backward widening.** A step that does not move φ past the last progress mark + σ_f counts as a stall.

$$\sigma_b = \min\big(\sigma_f\,(1 + \text{stall}/T(\varphi)),\ 0.5\big),\qquad T(\varphi) = \beta\textstyle\sum_j e^{-(\varphi_j-\varphi)^2/2\sigma_f^2}$$

T(φ) is the datapoints' share of phase φ, and β = 1 is the only tuning knob. While the arm is stuck:

1. φ freezes.
2. W fills the frozen slice until it matches Ŵ there, so the law sends the arm to the parts of the
   target it has not covered yet.
3. σ_b widens, so the target takes in **earlier-phase** datapoints: the arm backs off and tries other
   routes.

Once a route lets the projection advance, φ moves on and the stall counter resets. No step-back, goal
test or success detector is involved.

**Exact evaluation.** Both Ŵ and W are weighted sums of rank-1 Φ(z_p), so b is computed as
Σ_p a_p ⟨Λ, Φ(z_p) ∘ ∇Φ(x)⟩: one matrix chain through Λ's TT cores per point, with a = [−w; v/Σv]. There
is no per-step TT-cross and no K^d tensor. Points weighing below 10⁻⁶ of the heaviest are dropped.

**Loop.** This uses the unchanged `explore` loop:

- The torque goes out at 100 Hz and the law steps at 20 Hz, with U_MAX = 3 cube units/s.
- The commanded pose walks the straight line between setpoints, capped at 0.02 m/s.
- The Cartesian impedance gains are `make_controller`'s, with friction feedforward.
- The torque check refuses anything beyond ±8·b·c, and Ctrl-C hands the arm to the exit hold.
- Every 20th step prints φ, σ_b and stall/T.

## 5. The successful runs

Three runs on 2026-10-06 used `datapoints_demo` (168 points). All three reached the end of the task:

| run | φ ≥ 0.3 | φ ≥ 0.5 | φ ≥ 0.7 | φ ≥ 0.98 | σ_b at ceiling near φ | max \|τ\| joint 2 |
|---|---|---|---|---|---|---|
| `125449` | 23.1 s | 27.4 s | 65.8 s | 69.7 s | 0.30, 0.70 | 7.96 N·m |
| `125811` | 5.7 s | 32.7 s | 60.0 s | 63.5 s | 0.31, 0.70 | 8.00 N·m |
| `130046` | 16.5 s | 33.8 s | 65.0 s | 68.4 s | 0.30, 0.33, 0.70 | 7.84 N·m |

Run `125811`, in 10-s windows (peg tip, mm):

| t | φ | what the arm does |
|---|---|---|
| 0–10 s | 0 → 0.31 | +x along the bottom, x 201–295 |
| 10–20 s | 0.31 (stall) | σ_b reaches 0.5. It explores back over the start region, x 200–307, y down to −172 |
| 20–40 s | 0.32 → 0.70 | it goes up the **left** lane, x 215–242, to y ≈ 65. φ advances, because the projection is by position only |
| 40–60 s | 0.70 (stall) | it cannot continue +x from the left lane. σ_b reaches 0.5, it backs down to y ≈ −88 and moves over to x ≈ 296–300 |
| 60–70 s | 0.70 → 1.00 | up the **right** lane and +x to the goal at (378, 68) |

The other two runs follow the same pattern: a stall at the corner at φ ≈ 0.3, then the left lane, then a
stall at φ ≈ 0.7, then recovery through the right lane. This is the behaviour the method was designed
for. Nothing detects the dead end: the frozen phase, the filling statistic and the widening backward
kernel turn the arm around.

**With the first set (136 points)**, 15 runs were logged between 11:00 and 11:54. Only one (`114749`)
reached φ = 1, at 101.5 s. Eight of them stalled with φ between 0.83 and 0.87. My reading is that with
only 6 datapoints above φ = 0.8, the normalised target kept almost all its mass behind the arm, so the
forward pull at the last corner was too weak. The 32 extra datapoints at the end are what changed
between the failing and the succeeding runs. This is an inference from the decile counts and the stall
phases; I have not tested it in isolation.

## 6. Assessment

**What works**

- The full chain runs on the arm: master → labels → phase-conditioned E2T2 → Cartesian impedance.
  It solved an obstructed sequential task 3/3 with the demo set, in 63–70 s.
- Recovery from a dead end comes out of the method itself, with no goal threshold, step-back rule or
  operator input.
- The torques stay well inside the limits: |τ| ≤ 8.0 N·m on joint 2, against ±32 N·m.
- Offline checks pass: `test_phase_projection.py` 7/7 and `test_phase_ergodic_controller.py` 5/5, run
  today in the container.

**Weaknesses and open points**

1. **The phase clock follows the obstructed lane.** It projects by position only, so going up the left
   lane advanced φ from 0.32 to 0.70 although that lane is a dead end. Recovery worked only because the
   stall at 0.70 widened σ_b to its 0.5 ceiling, far enough back to re-include the right lane's
   datapoints at φ ≈ 0.3–0.5. With a later dead end, or a smaller ceiling, the arm would stay stuck.
   Orientation is also absent from the phase estimate (`summary.md`).
2. **The result depends on how evenly the datapoints cover φ.** A sparse final phase stalled the
   136-point set at φ ≈ 0.85. Nothing in the pipeline warns about this. The labeller prints the decile
   counts, but nothing checks them.
3. **The run does not stop at the goal.** After φ = 1 the law keeps exploring: `130046` reached
   φ = 0.99 at 68 s and had wandered back to (296, −84) by the time it was stopped at 84 s. Ending a run
   is up to the operator's Ctrl-C.
4. **Tracking error is large.** The 95th percentile of the distance from the peg tip to the commanded
   position was 40–47 mm in the three runs. It is probably dominated by contact with the obstacles, but
   I have not separated contact from lag.
5. **Evidence is thin.** There are three successes, with one datapoint set, one start pose and no
   repeated trials across sessions. The earlier attempts also changed parameters between runs: LEAD_SCALE
   went from 1 to 2 in `1125d88` and has since been set back to 1, and the planar restriction came in with `41e0644`.
6. **Memory grows without bound while stalled.** A step costs 4.6 ms at 2000 states (`summary.md`). The
   longest run, 106 s, is about 2100 steps, so this was not yet a limit.
7. **Docs lag behind the code.** `summary.md` describes a 6-D run, and the CLAUDE.md §5
   tree does not list `phase_projection.py` or `label_datapoint_phases.py`.

## 7. Reproducing the analysis

The numbers above come from the saved files and need no arm:

```bash
docker exec piper-ergodic bash -lc 'cd /home/jens/workspace/docker_intern_PiPER_ergodic/workspace/src && python -c "
import numpy as np
for ts in [\"125449\", \"125811\", \"130046\"]:
    tr = np.load(f\"../output/phase_trace_20261006_{ts}.npz\")
    i = np.argmax(tr[\"phi\"] >= 0.98); print(ts, tr[\"t\"][i], tr[\"phi\"][-1])"'
```

The runs and phase traces themselves are in the gitignored `workspace/output/`. Only the master and
datapoint sets are tracked.
