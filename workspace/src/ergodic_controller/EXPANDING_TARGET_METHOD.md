# The expanding-target ergodic law

What `ExpandingErgodicController` and `run_ergodic_expanding*.py` compute, as the
code stands at commit `c34ae67`. Every formula below is the code's; where the code
and this text ever disagree, the code is right and this file is stale.

Numbers marked *(2d_key)* are for the task in `saved_recordings/*/2d_key`.

| part | file |
|---|---|
| the law | `ergodic_controller/phase_ergodic_controller.py`, class `ExpandingErgodicController` |
| phase labels, band, entry widening | `direct_teaching/distribution/phase_projection.py` |
| task construction, constants | `run_ergodic_expanding.py` |
| trials | `run_ergodic_expanding_trials.py` |
| loop, gains, rates | `execution/live_ergodic_controller.py` |
| the 2-D simulation it was developed in | `ergodic_controller/Ergodic_Exploration_expanding_2D.ipynb` |

---

## 1. In one paragraph

A master trajectory defines a scalar progress variable, the phase $\varphi \in [0,1]$.
Every taught datapoint carries the phase of the master sample nearest to it. Online,
$\varphi$ is the furthest the peg has come along the master and never decreases. The
ergodic law (E2T2, unchanged) is given a target that consists of every datapoint
whose phase has been reached, plus a little ahead, and a time-average statistic that
slowly forgets. Nothing else: no stall counter, no per-stage goal detector, no
switching logic. Progress enlarges the target; the newly added slice is the only part
not yet covered, which is what pulls the arm forward.

---

## 2. Inputs

1. **Master**: one continuous demonstration of the whole task, `joint_angles_*.npz`,
   recorded at 100 Hz, stationary ends trimmed. Its peg-tip positions are
   $\bar{\mathbf p}_n \in \mathbb{R}^3$, $n = 1..N$ (FK of frame `peg_tcp`).
2. **Exploration datapoints**: $N_\mathrm{p}$ operator-confirmed poses,
   `datapoints_*_phase.npz`, each a joint vector with a phase label (§3.2).
   *(2d_key: $N_\mathrm{p} = 210$.)*

---

## 3. Offline construction (before the arm is touched)

### 3.1 The cube

The datapoints' poses are mapped to $[p, \operatorname{Log}_\mu(\text{quat})] \in \mathbb{R}^6$
and each axis is scaled to $[0,1]$:

$$\text{lower}/\text{upper} = \tfrac{s_{\min} + s_{\max}}{2} \mp (0.5 + m)\,(s_{\max} - s_{\min}), \qquad x = \frac{s - \text{lower}}{\text{upper} - \text{lower}},$$

with margin $m = 0.15$ (`CUBE_MARGIN`; the other runs use 0.1). The bounds come from
the original datapoints only, before the additions of §3.4. The law runs on the first
two cube axes, world $x$ and $y$, so in what follows a state is $\mathbf{x} \in [0,1]^2$
and the map to metres is linear, $p_{xy} = \text{lower}_{xy} + \mathbf{x} \odot \text{span}$.
*(2d_key: the cube is $0.429 \times 0.278$ m.)*

### 3.2 Phase of the master and of the datapoints

The master's phase is its normalised arc length,

$$\bar\varphi_n = \frac{\sum_{i=2}^{n} \lVert \bar{\mathbf p}_i - \bar{\mathbf p}_{i-1} \rVert}{\sum_{i=2}^{N} \lVert \bar{\mathbf p}_i - \bar{\mathbf p}_{i-1} \rVert},$$

so a pause in the demonstration earns no phase. The length is in 3-D, and with the
label file's `rotation_length = 0` orientation plays no part.
*(2d_key: length $S = 0.601$ m.)*

Datapoint $m$ gets the phase of its nearest master sample by peg-tip position in
3-D, $\varphi_m = \bar\varphi_{n^*(m)}$ (`label_datapoint_phases.py`, the first sample
on a tie).

### 3.3 The kernel width

$$\sigma_\mathrm{f} = \frac{h}{S}, \qquad h = \operatorname{median}_m \big(\text{distance from datapoint } m \text{ to its } k\text{-th nearest datapoint}\big), \quad k = \operatorname{round}(\sqrt{N_\mathrm{p}}).$$

$h$ is the datapoints' spatial resolution; divided by the master's length it is that
resolution as a phase. *(2d_key: $k = 14$, $h = 21.9$ mm, $\sigma_\mathrm{f} = 0.0364$.)*

### 3.4 Additions to the target points (`run_ergodic_expanding.py`)

Applied in this order; neither changes the law.

**Wider entry** (`with_wider_entry`). Let $\mathbf{o}$ be the master's $x$-$y$ position
at phase `ENTRY_PHASE` $= 0.2$ and $\hat{\mathbf n}$ the unit normal to the master's
chord from there to phase $0.25$. Every datapoint with $\varphi_m < 0.2$ is moved to

$$\mathbf p_m' = \mathbf p_m + \hat{\mathbf n}\; \ell_m \frac{d_\mathrm{e}}{\max_j |\ell_j|}, \qquad \ell_m = (\mathbf p_m - \mathbf o) \cdot \hat{\mathbf n},$$

with $d_\mathrm{e} = 0.01$ m (`ENTRY_WIDEN`). A proportional stretch across the way in:
the outermost point moves exactly $d_\mathrm{e}$, positions along the way in are kept.
*(2d_key: 69 datapoints; $y$ range $[-117.6, -30.3]$ mm becomes $[-126.3, -20.3]$ mm.)*

**Band beside the master** (`with_corridor`). The master's $x$-$y$ path is resampled
at stations every `CORRIDOR_SPACING` $= 0.01$ m of planar arc length. At each station
$\mathbf c_s$, with unit normal $\hat{\mathbf n}_s$ taken from the station-to-station
gradient, three points are added,

$$\mathbf c_s - d_\mathrm{b} \hat{\mathbf n}_s, \qquad \mathbf c_s, \qquad \mathbf c_s + d_\mathrm{b} \hat{\mathbf n}_s, \qquad d_\mathrm{b} = 0.02\ \text{m (`CORRIDOR_OFFSET`)},$$

each with the phase of its station (interpolated $\bar\varphi$) and treated from then
on as an ordinary datapoint. *(2d_key: 57 stations, 171 points; the target has
$M = 381$ points, of which the band is 45 %.)*

After both, all points are clipped to $[0,1]^2$ *(2d_key: none is clipped, and none
lies in the outer 5 % of the cube)*.

Call the resulting target points $\mathbf z_m \in [0,1]^2$ with phases $\varphi_m$,
$m = 1..M$.

---

## 4. The law, one ergodic step

Executed every $\Delta t = 0.05$ s (20 Hz) with the measured state $\mathbf x$: FK of
the measured joint angles, mapped to the cube, clipped to $[0,1]$, first two axes.

### 4.1 Phase update

$$\hat\varphi = \bar\varphi_{n^*}, \quad n^* = \arg\min_{n:\ \bar\varphi_n \in [\varphi - 0.05,\ \varphi + 0.12]} \big\lVert (\bar{\mathbf x}_n - \mathbf x) \odot \text{span} \big\rVert, \qquad \varphi \leftarrow \max(\varphi, \hat\varphi).$$

The distance is in metres, in $x$-$y$. The window is clipped to $[0,1]$. $\varphi$
starts at 0.

### 4.2 Memory

$\mathbf x$ is appended to the list of past states $\mathbf x_1, \dots, \mathbf x_\ell$.
One state per ergodic step, including the step taken before the loop starts.

### 4.3 Target weights

With $\delta_m = \varphi_m - \min(1,\ \varphi + \sigma_\mathrm{f})$:

$$f_m = \exp\!\Big(\!-\frac{\delta_m^2}{2\sigma_\mathrm{f}^2}\Big), \qquad r_m = \begin{cases} 1 & \delta_m \le 0 \\ f_m & \delta_m > 0 \end{cases}, \qquad \hat w_m = (1-\eta)\,\frac{r_m}{\sum_j r_j} + \eta\,\frac{f_m}{\sum_j f_j},$$

$\eta = 0.25$ (`FRONT_SHARE`). $r$ is "reached": weight 1 for every point at or
behind the front $\varphi + \sigma_\mathrm{f}$, a Gaussian tail of width
$\sigma_\mathrm{f}$ ahead of it. $f$ is the newest slice alone. $\sum_m \hat w_m = 1$.

### 4.4 Statistic weights

$$v_{\ell'} = \frac{\lambda^{\ell - \ell'}}{\sum_{j=1}^{\ell} \lambda^{\ell - j}}, \qquad \lambda = 1 - \frac{1}{n_\mathrm{f}}, \qquad n_\mathrm{f} = 75 \cdot \frac{h}{v_{\max}\,\Delta t},$$

with $v_{\max} = 0.02$ m/s (`MAX_SPEED`) and 75 = `FORGET_PROGRESS`. $h/(v_{\max}\Delta t)$
is the number of ergodic steps one $\sigma_\mathrm{f}$ of progress takes at the speed
cap. A state $n_\mathrm{f}$ steps old weighs $1/e$ of the newest.
*(2d_key: 21.9 steps, $n_\mathrm{f} = 1641$ steps $= 82$ s, $\lambda = 0.999391$.)*

### 4.5 Direction

With the cosine basis $\Phi_{\mathbf k}(\mathbf x) = \cos(\pi k_1 x_1)\cos(\pi k_2 x_2)$,
$k_i \in \{0,\dots,9\}$ ($K = 10$), and mode weights
$\Lambda_{\mathbf k} = (1 + \lVert \mathbf k \rVert^2)^{-3/2}$ (the E2T2 weights for
$d = 2$, held as a tensor train rounded to $10^{-2}$):

$$\mathcal W_{\mathbf k} = \sum_{\ell'} v_{\ell'}\, \Phi_{\mathbf k}(\mathbf x_{\ell'}), \qquad \hat{\mathcal W}_{\mathbf k} = \sum_m \hat w_m\, \Phi_{\mathbf k}(\mathbf z_m), \qquad b_i = \sum_{\mathbf k} \Lambda_{\mathbf k}\,\big(\mathcal W_{\mathbf k} - \hat{\mathcal W}_{\mathbf k}\big)\, \frac{\partial \Phi_{\mathbf k}}{\partial x_i}(\mathbf x).$$

$\mathbf b$ is proportional to the gradient, with respect to the newest state, of the
ergodic metric $\sum_{\mathbf k} \Lambda_{\mathbf k} (\mathcal W_{\mathbf k} - \hat{\mathcal W}_{\mathbf k})^2$
(the factor is twice that state's weight). It is evaluated exactly, as a sum over the
points $\mathbf z_m$ and $\mathbf x_{\ell'}$ with one matrix chain through $\Lambda$'s
TT cores per point; no density is fitted and no coefficients are precomputed.

$$\mathbf u_\mathrm{e} = -\frac{\mathbf b}{\lVert \mathbf b \rVert + 10^{-10}}.$$

### 4.6 Boundary pull and output

Per axis, with $\alpha = 20$, $c = 0.05$:

$$w_i = \tfrac12\tanh\!\big(\alpha(x_i - c)\big) + \tfrac12\tanh\!\big(\alpha(1 - c - x_i)\big), \qquad g_i = -\tfrac12\tanh\!\big(\alpha(x_i - c)\big) + \tfrac12\tanh\!\big(\alpha(1 - c - x_i)\big),$$

$$\mathbf u = \mathbf u_\mathrm{e} \odot \mathbf w + \frac{\mathbf g}{\lVert \mathbf g \rVert + 10^{-8}} \odot (1 - \mathbf w), \qquad \mathbf x^{+} = \operatorname{clip}\!\Big(\mathbf x + \Delta t\, u_{\max} \frac{\mathbf u}{\lVert \mathbf u \rVert + 10^{-8}},\ 0,\ 1\Big),$$

$u_{\max} = 3$ cube units/s. $w_i \approx 1$ in the interior and falls to 0 in the
outer twentieth of the cube, where the output is replaced by a push inwards.
$\mathbf x^{+}$ is the law's entire output: a setpoint 0.15 cube units from the
measured state.

### 4.7 What the law does not use

The class inherits a stall counter, $\sigma_\mathrm{b}$, a dwell $T(\varphi)$ and an
on-reference test from `PhaseErgodicController`. They still run and fill the trace's
second and third columns, but nothing in §4.3 to §4.6 reads them.

---

## 5. From setpoint to torque

Unchanged from the other runs (`execution/live_ergodic_controller.py`).

- **Interpolation.** The commanded state $\tilde{\mathbf x}$ walks the straight line
  towards $\mathbf x^{+}$ in 5 control cycles, but its peg-tip speed is capped at
  $v_{\max} = 0.02$ m/s. Since 0.15 cube units is several centimetres, the cap always
  binds: the commanded pose moves 1 mm per ergodic step in the direction the law
  gave, and the next interval starts from where $\tilde{\mathbf x}$ actually is.
- **Planar.** Only $x$-$y$ come from the law. $z$ and the peg's tilt are the start
  pose's for the whole run; rotation about the peg axis is neither held nor driven,
  and joint 6 has a soft limit.
- **Torque**, at 100 Hz: Cartesian impedance on `peg_tcp` with
  $k = [250, 250, 300, 3, 3, 1]$, $b = [5, 5, 5, 0.2, 0.2, 0.1]$, joint torque weights
  $[1, 1, 1, 0.5, 0.5, 0.5]$, plus gravity/Coriolis compensation and the friction
  feedforward. A torque beyond the firmware limit raises and the arm is held.

So the arm is a spring-damper pulled at 20 mm/s along the law's direction. Against an
obstacle the commanded pose runs ahead of the peg, the measured state stops moving,
and the law sees that through §4.1 and §4.2.

---

## 6. A run and a trial

- **Start.** Place the arm near the master's start. $\varphi = 0$, memory empty.
- **Stop.** The peg tip within `GOAL_RADIUS` $= 0.01$ m of the master's last peg-tip
  position, in $x$-$y$; or Ctrl-C. Either way the arm goes to a position hold first.
- **Trials** (`run_ergodic_expanding_trials.py`). Between trials the arm is
  backdrivable and the law is not stepped. Each trial starts with $\varphi = 0$ and
  an **empty memory**. Trials share nothing, so they are independent repeats.

---

## 7. Assumptions

About the task:

1. **Progress is one-dimensional and monotone.** The task can be ordered along one
   demonstrated path, and it never requires going back to an earlier phase *as a
   goal*. (Going back while searching is fine; the phase just does not decrease.)
2. **Position in the plane determines progress.** The phase is read from $x$-$y$
   position alone. Two task states at the same $x$-$y$ but different orientation or
   height are the same phase. A master that crosses or closely passes itself can be
   mis-read, limited only by the window $[\varphi - 0.05, \varphi + 0.12]$.
3. **The task is planar** as run here: $z$ and tilt are constants of the start pose.
4. **The goal is the master's end**, reachable to within 10 mm.

About the taught data:

5. **The datapoints cover the ways through the task**, including any alternative the
   arm may need. The law explores only where target points are; it does not invent
   a route nobody taught, beyond the 20 mm band.
6. **Phase labels are meaningful**: a datapoint's nearest master sample is the stage
   it belongs to. A datapoint far from the master gets the phase of wherever the
   master happens to be closest.
7. **The master was recorded on the same geometry the run meets**, at least in where
   it ends. In the 2-D simulation a demonstration that turned from the wrong angle
   froze the key.
8. **`ENTRY_PHASE = 0.2` is where the search ends** on this master. It is a constant
   of 2d_key, not derived.

About the arm and the law:

9. **The arm can follow at 20 mm/s** well enough that the measured state reflects
   what the law asked for, and an obstacle shows up as lack of motion, not as a
   torque fault.
10. **$K = 10$ modes resolve what matters.** The finest mode has a half-wavelength of
    span/9, about 3 to 5 cm on 2d_key. Structure finer than that (the 2 cm band, the
    1 cm widening) is seen blurred.
11. **The forgetting window is long against a pass and short against a stall**:
    $n_\mathrm{f} = 82$ s on 2d_key, against a 34 s free run.
12. **$\eta$, the factor 75, the band and the widening are not derived.** They come
    from a 2-D simulation and from tuning on one task.

---

## 8. Why this works on a sequential problem

A sequential task is hard for plain ergodic control because the target density is
all stages at once: the law has no reason to do stage 1 before stage 3, and time
spent in stage 3's region counts as coverage whether or not stage 1 was achieved.

**Order comes from the phase gate.** Only points with $\varphi_m \lesssim \varphi + \sigma_\mathrm{f}$
have weight (§4.3). At $\varphi = 0$ the target is the entry cluster alone. Stage 2's
points do not exist for the law until the peg has physically got to where stage 1
ends, because $\varphi$ only moves when the *measured* position projects further along
the master (§4.1). Achieving a stage is therefore detected by the same quantity that
defines the stages, with no per-stage success test.

**The forward pull is the uncovered slice.** When $\varphi$ advances, a slice of
points joins the target. The statistic has no mass there yet, so
$\mathcal W - \hat{\mathcal W}$ is most negative there and $-\mathbf b$ points into it.
The front share $\eta$ strengthens this: without it the new slice is a shrinking
fraction of an ever larger target (in the 2-D simulation, 15/35 with $\eta = 0$ against
29/35 with $\eta = 0.25$).

**Nothing reached is lost.** $r_m$ never decreases as $\varphi$ grows, and $\varphi$
never decreases. So when the arm is blocked, the target is still everything reached
so far, including every taught alternative at earlier phases. The law keeps covering
that set, which is the search for another way, without any stall detection deciding
to start it.

**Being stuck resolves itself through coverage.** A blocked peg accumulates
statistic where it stands. That place becomes over-covered, $-\mathbf b$ points away
from it, and the arm leaves to cover the rest of the reached set. Forgetting (§4.4)
makes this repeatable: an over-covered place fades over $n_\mathrm{f}$ steps and is
tried again, instead of being ruled out for the rest of the run by one long visit.

**Any route that gets further counts.** The phase is a projection onto the master,
not a requirement to be on it. If the arm gets past a blockage by a taught detour and
comes out further along, $\varphi$ jumps (by at most 0.12 per step) and the next
slice opens.

The limits follow from the same mechanism. If no target point lies on a workable
detour, the law covers the reached set forever. If a detour runs parallel to the
master, the phase advances on it although the task may not have (assumption 2).

---

## 9. Ways to verify it

### 9.1 That the code computes §4 (offline, exact)

Run from `workspace/src` in the container; all are in
`ergodic_controller/test_phase_ergodic_controller.py` unless noted.

| claim | test | compared against |
|---|---|---|
| $\mathbf b$ is the TT expression | `test_point_sum_gradient_matches_tt_dot` | `tt.dot` of the summed TT with the basis gradient, to $10^{-10}$ |
| with one phase and no forgetting it is E2T2 | `test_expanding_law_on_a_frozen_phase_is_the_phase_law`, with `test_frozen_phase_steps_as_ergodic_controller` | `ErgodicController`, step by step, to $10^{-12}$ |
| target weights are §4.3 and only grow | `test_expanding_target_holds_everything_reached_and_only_grows` | the formula written out per datapoint |
| statistic is §4.4 | `test_expanding_statistic_is_the_discounted_time_average` | the recursion $W \leftarrow \lambda W + \Phi(\mathbf x)$, $n \leftarrow \lambda n + 1$ |
| phase is monotone and windowed | `test_phase_estimate_is_monotone_and_windowed` | hand-placed states |
| closed loop reaches the end | `test_expanding_law_explores_a_straight_task_to_its_end` | phase $\ge 0.99$; $\eta = 0$ covers both sides, $\eta = 0.25$ is faster |
| band geometry | `test_corridor_points_*` in `test_phase_projection.py` | a quarter circle: radii $r$, $r \pm d_\mathrm{b}$ |
| entry widening | `test_widen_across_*` in `test_phase_projection.py` | $y$ scaled analytically, and its rotation |

```bash
docker exec piper-ergodic bash -lc 'cd /home/jens/workspace/docker_intern_PiPER_ergodic/workspace/src && python -c "
import ergodic_controller.test_phase_ergodic_controller as t
[getattr(t, n)() for n in dir(t) if n.startswith(\"test_\")]; print(\"ok\")"'
```

### 9.2 That the task is built as §3 says (offline, on the real files)

- Map the target points back to metres and measure each band point's distance to the
  master from forward kinematics. *(2d_key, at the 20 mm offset: side rows
  median 19.7 mm, maximum 20.0 mm, and as close as 10 mm on the inside of a corner;
  centre row within 1.0 mm.)*
- Compare the entry datapoints before and after widening. *(2d_key: outermost point
  moved exactly 10.0 mm, all other datapoints unchanged.)*
- Count points in the outer 5 % of the cube; it should be zero.

### 9.3 That the whole script runs and exits safely (offline, fake arm)

`simulation/fake_executor_helpers.install(q, max_cycles=...)` **before** importing
the script, then call its `main`. The fake arm never moves, so this checks the
plumbing, the goal stop and the exit hold only: started at the master's last pose it
must stop after one torque; started at the first it must run until interrupted and
report the arm held.

### 9.4 That the mechanism of §8 is what happens (on the arm, from saved files)

From `ergodic_run_*_expanding*.npz` and `phase_trace_*.npz`:

- **Phase gate.** At every step, the furthest target point with non-negligible
  weight must be within about $2\sigma_\mathrm{f}$ of $\varphi$. Equivalent check on a
  run: the peg is never commanded into a later leg before $\varphi$ gets there.
- **Monotone phase**: `np.diff(phi) >= 0` within a trial.
- **Blocked means not moving, not faulting**: during a phase plateau, the tracking
  error (commanded minus measured peg tip) is large and torques stay below the limit.
- **Leaving a blockage**: after a plateau, the peg's distance from the place it was
  stuck grows before $\varphi$ next advances.
- **Repeatability**: the same start pose gives the same time.

### 9.5 That it is better than the alternative (not done on the arm)

These are the checks that would show the design choices matter. None has been run on
the arm.

- $\eta = 0$ against $0.25$, same task and start.
- `forget_window = None` against the 82 s window, on a task with a dead end.
- With and without the band; with and without the wider entry.
- Against `run_ergodic_phase.py` (the moving-kernel law) on the same task.

---

## 10. What the evidence is, and is not

| evidence | result | what it shows |
|---|---|---|
| 2-D key-in-lock simulation, 35 + 30 runs, $\eta = 0.25$ | expanding target alone 29/35 and 16/30; with a correctly taught turn and forgetting 30/35 and 25/30 | the mechanism, in a toy with idealised contact |
| 2d_key on the arm, 10 trials, sequence `20261009_100213` | 10/10 reached the goal; trial 1 in 70.6 s with a 31 s plateau at $\varphi = 0.39$ under contact; trials 2 to 10 in $34.2 \pm 0.4$ s | it completes this task, recovers from one blockage, and is repeatable |
| 2d_key, 8 trials with stalled paths carried over (removed) | 34 s, then 108 s, 193 s, then never left the entry | memory across trials of *where it stalled* covers the way in; not part of the method |

Not shown: robustness to a different start pose, a second task, a task blocked in
every trial, or any of the ablations in §9.5. The offline band comparison (ideal
tracking, no contact) found the band does not increase time spent 5 to 15 mm from the
master; on two_gaps it pulled the path *towards* the master.
