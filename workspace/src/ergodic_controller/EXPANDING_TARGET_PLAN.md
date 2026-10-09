# Plan: the expanding-target law in 6 DoF

How to take the method of `EXPANDING_TARGET_METHOD.md` from the planar task to a
full-pose task (peg-in-hole style), in its own script
`run_ergodic_expansion_6dof.py`. Measurements quoted are from an offline check on
`saved_recordings/*/peg_in_hole`.

**Status.** §3 is implemented with the recommendations of §4: no band in 6-D, an
angular cap `MAX_ANGULAR_SPEED = 0.33` rad/s in `plan_interval`, the position-only
goal, and no joint-limit guard. Steps 1 to 4 of §6 have been run; 5 to 7 need the
arm. Step 3 found that 131 of the 165 peg_in_hole datapoints were taught with joint 6
beyond its URDF limit of 2.094 rad (up to 3.86 rad), so the offline joint-limit check
cannot pass on that set for joint 6; joints 1 to 5 keep at least 0.25 rad of margin.

---

## 1. What carries over unchanged

**The law.** `ExpandingErgodicController` has no planar code in it. The number of
axes comes from `PhaseTask.axes`; the target weights, the forgetting statistic and
the gradient are the same formulas on $[0,1]^6$. Checked offline: built on the
peg_in_hole task with `axes = 6`, it steps without error and the phase runs to 1.

**The cost.** A 6-D step is cheap, because $\Lambda$'s tensor train has rank 2 and
the gradient is one matrix chain per point:

| past states in the statistic | one step |
|---|---|
| 200 | 0.5 ms |
| 1000 | 0.7 ms |
| 3000 | 1.1 ms |
| 6000 (5 min at 20 Hz) | 1.7 ms |

against a 10 ms control cycle. No cap on the memory is needed for runs of this length.

**The task builder.** `prepare_phase_law(master, labelled, planar=False,
track_master=True, margin=...)` already builds the 6-D task: all six cube axes, and
the phase projected on the full pose with the metric
$d^2 = \lVert \Delta p \rVert^2 + (l\,\theta)^2$, $\theta$ the geodesic rotation angle
and $l$ the `rotation_length` stored in the label file.

**The loop.** `explore`, the gains, the interpolation and the goal test
(`reached_goal`, which uses $x$-$y$-$z$ when not planar) already handle
`planar=False`; `run_ergodic_phase_6dof.py` runs through them today.

---

## 2. What does not carry over

| planar piece | why not | plan |
|---|---|---|
| band beside the master (`corridor_points`) | defined by a normal in the $x$-$y$ plane; in 3-D the master has a normal *plane*, and orientation has no "beside" | leave out of the first version (§4, decision A) |
| wider entry (`widen_across`, `ENTRY_PHASE`) | same planar geometry, and the constant 0.2 belongs to 2d_key | leave out; widen by teaching more datapoints |
| planar constraints (fixed $z$ and tilt, free peg rotation, joint 6 soft limit) | the law now commands all six axes | already switched off by `planar=False` |
| forgetting window $n_\mathrm{f} = 75\,h/(v_{\max}\Delta t)$ | $h$ is now a pose distance including $l\theta$, but $v_{\max}$ caps linear speed only | keep the formula for the first runs, treat 75 as the knob (§5.3) |

The offline band comparison found that the band does not increase the time spent
beside the master, so dropping it costs nothing that has been measured.

---

## 3. Implementation

Three steps, in this order. Estimated size in lines of diff.

### 3.1 Make `run_ergodic_expanding.py` take `planar` (~15 lines)

```python
def prepare_expanding_exploration(master, labelled, planar: bool = True) -> Exploration:
    controller, distribution, phase_law, master_ends = prepare_phase_law(
        master, labelled, planar=planar, track_master=True, margin=CUBE_MARGIN
    )
    task = phase_law.task
    if planar:
        task = with_corridor(with_wider_entry(task, X_master), X_master)
    ...
    return Exploration(..., planar=planar)

def main(master, labelled, planar: bool = True) -> None:
    ...
    save_run(log, labelled, label="_expanding" if planar else "_expanding6")
```

The planar path must stay bit-identical: same task, same constants, same file names.

### 3.2 `run_ergodic_expansion_6dof.py` (~30 lines, a wrapper)

The same shape as `run_ergodic_phase_6dof.py`: a docstring, the two mandatory
arguments, and `main(args.master, args.labelled, planar=False)`. No law, gain or
constant lives in it, so the two expanding runs cannot drift apart.

It touches the arm, so it goes on the list in `CLAUDE.md` §1 and §5, with the two
planar expanding scripts that are missing there too.

### 3.3 A 6-D test (~25 lines, in `test_phase_ergodic_controller.py`)

- Target weights on a 6-D task against the formula written out (the existing test is
  2-D).
- Closed loop on a 6-D straight-line master whose orientation also turns: the phase
  reaches the end, which exercises the rotation term of the phase estimate together
  with the expanding target.

Not in this plan: a 6-DoF trials script. It is the same wrapper pattern over
`run_ergodic_expanding_trials.py` once that takes `planar`, and is worth writing only
after a single 6-DoF run works.

---

## 4. Decisions needed before writing it

**A. Band in 6-D: none, or a ring?** Recommended: none. The alternative is a ring of
points around each station in the plane normal to the master's tangent, at the
station's orientation. It triples or more the target's point count in directions
nobody taught, including into the fixture.

**B. A cap on rotation speed.** `plan_interval` caps the commanded *linear* speed at
`MAX_SPEED` and scales the whole step by that factor. A setpoint that is almost pure
rotation is therefore not capped at all. One law step is 0.15 cube units; on
peg_in_hole the third orientation axis spans 2.27 half-angle rad, so a step along it
alone asks for about 0.68 rad of peg rotation in 50 ms. The existing 6-DoF phase run
has the same exposure. Recommended: add an angular cap to `plan_interval` (one
constant, the scale becomes the smaller of the two), as its own change, tested on the
planar run first where it must change nothing. This is a change to the shared loop,
so it needs your go-ahead.

**C. Goal test.** Position only, within 10 mm of the master's last peg-tip position
in $x$-$y$-$z$, as the other 6-DoF trials use. For an insertion that is depth along
the hole, which is what matters; orientation is not checked. Recommended: keep it,
and add an orientation tolerance only if a run ends "at the goal" visibly tilted.

**D. Joint limits.** The planar run cannot leave the taught workspace by much. A
6-D target can ask for orientations near a joint limit, and no loop watches $q$
against the limits (`CLAUDE.md` §8). The torque guard still holds the arm if a torque
exceeds the firmware range. Recommended for the first runs: teach datapoints well
inside the limits and watch joints 4 to 6; decide on a guard after seeing them.

---

## 5. What the data must provide

### 5.1 Label with a rotation length

```bash
python label_datapoint_phases.py master.npz datapoints.npz --rotation-length 0.06
```

With $l = 0$ the phase ignores orientation, and a stage that is a pure rotation (a
key turning in place) earns no phase at all. $l$ is metres per radian: 0.06 makes
1 rad of rotation count as 60 mm of travel. `peg_in_hole` is labelled with 0.06.

### 5.2 Enough datapoints for a usable $\sigma_\mathrm{f}$

$\sigma_\mathrm{f} = h / S$ is the datapoints' resolution as a phase. On peg_in_hole
(165 datapoints) it is **0.118**, against 0.036 on 2d_key. That is coarse: the front
of the target is 12 % of the task wide, about the size of the phase estimator's
look-ahead window (0.12), so the "newest slice" is a third of the task and the phase
gate orders the task only roughly. More datapoints, or datapoints closer together
along the master, bring it down. Aim for $\sigma_\mathrm{f} \lesssim 0.05$ before
judging the method on a 6-D task.

### 5.3 The forgetting window

With the present formula peg_in_hole gives $n_\mathrm{f} = 2236$ steps $= 112$ s. On
2d_key the window was 82 s against a 34 s free run, a ratio of about 2.4. Check the
same ratio after the first free 6-DoF run and adjust `FORGET_PROGRESS` to keep it,
since rotation makes the step estimate behind the formula unreliable (§2).

### 5.4 The master on the real geometry

Record the master on the fixture as it will be run, ending inserted. In the 2-D
simulation a demonstration whose final stage started from the wrong angle froze the
key against the lock's wall.

### 5.5 Spread at the entry, taught

Without the planar widening, the search before the hole is as wide as the datapoints
confirmed there. The cube's orientation axes are scaled by the datapoints' own range,
so tilt variation has to be taught too: poses at the hole's mouth at the offsets and
tilts the peg may arrive with.

---

## 6. Verification, in order

Stop at the first step that fails.

1. **Planar unchanged.** After §3.1, the planar task is identical: same number of
   target points (381 on 2d_key), same $\sigma_\mathrm{f}$ and $n_\mathrm{f}$, and the
   fake-arm run writes the same number of cycles. All existing tests pass.
2. **6-D tests** of §3.3 pass.
3. **Offline closed loop on the real 6-D files**, ideal tracking at the arm's step
   (1 mm per ergodic step, with the rotation cap of decision B if adopted): the phase
   reaches 1, every commanded state stays inside $[0.05, 0.95]^6$, and the commanded
   joint angles (closed-form IK of each commanded pose, `kinematics/kinematic_solver.py`)
   stay inside the joint limits with margin. This is the check for decision D that
   needs no arm.
4. **Fake arm.** `fake_executor_helpers.install(q, ...)` before importing the script:
   started at the master's last pose it stops after one torque; started at the first
   it runs until interrupted and reports the arm held. Largest torque printed.
5. **On the arm, free space.** You run it, with the fixture removed or the arm
   started clear of it, so the first 6-DoF motion meets nothing. Expect a run close to
   the master's own duration scaled by speed. Look at: rotation rate of the commanded
   pose, torques on joints 4 to 6, tracking error.
6. **On the arm, the task.** One run, then several from the same start.
7. **Compare** against `run_ergodic_phase_6dof.py` on the same files and start pose:
   time to the goal, and number of runs reaching it.

What to read from each saved run is in `EXPANDING_TARGET_METHOD.md` §9.4.

---

## 7. Risks

| risk | sign | response |
|---|---|---|
| fast rotation of the commanded pose | large torque on joints 4 to 6, a run ending on the torque guard | decision B |
| phase advances on rotation or parallel motion that is not progress | $\varphi$ reaches 1 away from the goal | raise or lower $l$; more datapoints near the master |
| coarse $\sigma_\mathrm{f}$ | the arm heads for the hole before it is aligned | §5.2 |
| target near a joint limit | a joint at its stop, tracking error on orientation | decision D, step 3 of §6 |
| orientation axes dominate the cube | the law spends its steps turning the peg, little translation | the cube axes are per-axis scaled, so this means the datapoints vary more in orientation than intended: re-teach |
| too few points for six dimensions | the path looks like the planar one with noise on the other axes | expected to a degree: $K = 10$ modes and a few hundred points give a smooth, low-detail density |

---

## 8. Open questions this plan does not answer

- Whether the expanding target beats the moving-kernel law in 6 DoF. The 2-D
  simulation and one planar task are the only evidence so far.
- Whether a jammed peg backs out by itself. In the planar runs a blocked peg left
  through coverage of the reached set; in a hole, leaving means moving back up, which
  needs taught datapoints above the hole at an earlier phase (they exist in
  peg_in_hole: 29 % of its datapoints have $\varphi < 0.2$).
- Trials in 6 DoF, and whether anything should be carried between them.
