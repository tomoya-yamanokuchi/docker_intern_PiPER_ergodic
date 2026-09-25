# CLAUDE.md — PiPER Ergodic / Cartesian Impedance Control

Guidance for Claude Code in this repository. It is the authoritative document;
`README.md` is a short human-facing intro and lags behind (it still describes a
tree holding only `agx_reference/`).

---

## 1. Hard rules

1. **Never run Python that can move the robot.** A real AgileX PiPER arm is
   physically connected. The scripts that open a session with it are
   `agx_reference/piper/main_*.py`, `record_joint_angles.py`,
   `play_joint_angles.py`, `identify_friction.py` and `run_ergodic_pipeline.py`
   — and, generally, anything that imports `pyAgxArm` or
   `execution.executor_helpers`, or calls `robot.connect()`, `robot.enable()`,
   `move_mit`, `move_p` or `move_j`. The user runs those; Claude writes the code
   and reads the output the user pastes back. Read-only inspection is fine:
   sources, the installed `pyAgxArm`, `ip link show can0`, `candump can0`.

   **`core/`, `controller/`, `direct_teaching/`, `ergodic_controller/`,
   `kinematics/`, `simulation/` and `visualization/` import no SDK and are safe to
   run**, so a torque law, an FK call, an ergodic step, a recording or an analysis
   of one can be checked offline — and `simulation/fake_executor_helpers.py` runs
   a whole execution script against a fake arm. Keep that split: a new control law
   or analysis goes in a hardware-free module; only a top-level script touches the
   arm.
2. **Never create git commits** unless asked, and never add `Co-Authored-By:
   Claude` or `Claude-Session:` trailers.
3. **`workspace/src/agx_reference/` is third-party code that stays close to
   upstream** (§4). Do not reformat it, restyle its Chinese comments, or "fix"
   its bare `except:` clauses. New work goes beside it, importing from it.
4. **Only `workspace/src/` is live code.** `double_PiPER/` is the vendor's ROS1
   package tree, kept as reference only. No ROS is used anywhere. Do not add to,
   build or "fix" it.

---

## 2. Hardware safety: the arm has no holding brakes

**The PiPER falls the instant it loses motor torque.** The vendor documents
this (`double_PiPER/README(EN).md` §2.2): stopping lets it fall with constant
damping, resetting makes it lose power and fall immediately.

Everything that removes torque drops it: cutting AC, the E-stop, disabling the
arm, and MIT mode with `kp = kd = t_ff = 0`.

Before a deliberate power-down, command the arm low and retracted if CAN works,
otherwise physically support it and clear the swing path. Power-cycle sequence:
support the arm, remove AC, wait ~10 s, restore AC, re-activate `can0` (a USB
re-plug leaves the interface DOWN with no bitrate, §6).

---

## 3. The project and where it stands

Direct, ROS-free control of an AgileX PiPER 6-DoF arm from Python over
SocketCAN, working towards an ergodic-exploration controller (E2T2,
tensor-train) for a peg-in-hole style contact task.

```
E2T2 / ergodic controller  ->  Python 3.10  ->  pyAgxArm
  ->  python-can  ->  socketcan  ->  can0 (gs_usb USB-CAN @ 1 Mbit/s)  ->  PiPER
```

**Milestone reached: the online peg-in-hole task.** The whole chain runs on the
arm — a demonstration is recorded under gravity compensation, a time-weighted
distribution is fitted over the poses it visited, the E2T2 ergodic law explores
that distribution, and the Cartesian impedance law tracks the pose it commands,
with the run watchable live in MeshCat. `run_ergodic_pipeline.py` is that chain
(§5); everything below it is validated on the arm.

The pieces, in the order the data flows:

- **Impedance control** — Cartesian (end effector as a spring-damper in task
  space, null space free), joint impedance, and pure gravity compensation. That
  is `agx_reference/` (§4).
- **Inertia and friction feedforward** — `controller/feed_forward.py`, on top of
  the impedance torque. A gravity-model correction, and friction that keeps a
  static level with a Stribeck decay gated on the *measured* speed, so a
  standing joint gets enough torque to break away. The constants are fitted from
  constant-velocity sweeps (`identify_friction.py`).
- **Direct teaching** — the operator backdrives the arm while `q` is recorded at
  the control-loop rate (`REC_FREQ_HZ = 100`); a recording can be replayed under
  the Cartesian impedance law.
- **The target distribution** — `direct_teaching/distribution/pose_distribution.py`.
  A time-weighted GMM, so a sample's weight is the interval it covers and a
  dwell is dense while a pass-through is sparse. **"Place" is the full 6-DoF
  pose**, as the E2T2 paper has it: the state is `[p, Log_mu(quat)]`, the
  orientation part a half-angle quaternion logarithm about the demonstration's
  mean orientation, scaled per axis into `[0, 1]^6`. The frame is `peg_tcp`, the
  peg tip 60 mm beyond `link6`, never `link6` itself.
- **The ergodic controller** — `ergodic_controller/ergodic_controller.py`, the
  E2T2 tensor-train law. Its output is a target pose and that pose is the entire
  interface to the impedance controller; the two stay separate components.
- **Closed-form IK** — `kinematics/kinematic_solver.py`, used offline only (the
  online loop needs no IK, since the ergodic target is a pose the impedance law
  tracks directly).
- **Views** — `visualization/visualizer.py`: offline review of a recording, a
  distribution or a finished run, plus `LiveView`, the live MeshCat scene of a
  run. `simulation/fake_executor_helpers.py` runs an execution script against a
  fake arm, so a loop can be exercised end to end with no hardware.

Two rates, deliberately different: the torque goes out at 100 Hz and the ergodic
law sets a setpoint at 20 Hz, with the commanded pose walking the line between
setpoints. Output goes in the gitignored `workspace/output/`.

**Next milestone: multiple trials, as in the E2T2 paper.** Between trials the
arm's pose is reset to a start pose while the ergodic controller's accumulated
history is kept, and the measurement is that the time to find the hole decreases
over the sequence of trials. The load-bearing part is that a reset moves the arm,
not the statistics: clearing the accumulated coverage at a trial boundary — or
building each trial on a fresh `ErgodicController` — destroys the very effect the
experiment exists to show.

The E2T2 reference is `~/Ergodic_Exploration_using_Tensor_Train` (two notebooks,
NumPy and JAX), mounted into the container (§7).

---

## 4. `agx_reference/` — the validated implementation

A copy of `kehuanjack/agilex-arm-gravity-compensation`, branch `imp`, checked
against upstream `HEAD` `8e2545c`, plus its own reduced 6-DoF URDF and meshes.
Eight Python files, ~1100 lines — six of them upstream's, two of them ours
(`feed_forward.py` and its test, marked below):

```
workspace/src/agx_reference/
├── core/agx_pinocchio.py              # Pinocchio wrapper: FK, Jacobian, nle, rnea
├── controller/
│   ├── task_imp_controller.py         # CartesianImpedanceController -> joint torques
│   ├── jnt_imp_controller.py          # JointImpedanceController     -> joint torques
│   ├── feed_forward.py                # OURS: gravity correction + inertia + friction
│   └── test_feed_forward.py           # OURS
└── piper/
    ├── main_gc.py                     # demo: pure gravity compensation
    ├── main_tast_imp.py               # demo: Cartesian impedance hold  [sic: "tast"]
    ├── main_jnt_imp.py                # demo: joint impedance hold
    └── piper/{urdf,meshes}/           # piper_description.urdf, .dae visual, .stl collision
```

`feed_forward.py` sits here rather than beside our own code because it is
imported as `controller.feed_forward`, the same import root as the two
controllers it is added to (§5). It is ours, so hard rule 3 does not protect it:
it is linted and refactored like the rest of our code.

**Deviation from upstream `HEAD`**, checked with
`diff <(git -C ~/agilex-arm-gravity-compensation show HEAD:<path>) <local>`:

- `agx_pinocchio.py` — byte-identical.
- `jnt_imp_controller.py` — whitespace only.
- `task_imp_controller.py` — whitespace, plus one commented-out alternative
  damping vector above its default `set_cart_params` call, left from tuning the
  ergodic run. Dead code in an upstream file; delete it rather than extend it.
- all three `main_*.py` — `PiperFW.DEFAULT` where upstream passes `PiperFW.V189`
  (this arm's firmware needs `DEFAULT`, §8), with the comment
  `# this arm reports S-V1.8-2` in `main_gc.py` and `main_tast_imp.py`.
- `main_tast_imp.py` **also** differs in substance: `control_frequency` 100.0
  where upstream has 200.0, `joint_torque_weights` `[1, 1, 1, 0.5, 0.5, 0.5]`
  where upstream has `[1, 1, 1, 0.5, 1, 0.5]`, `k` `[200, 100, 100, 5, 5, 5]`
  where upstream has `[200, 200, 200, 5, 5, 5]`, and two leftover `print(b)` /
  `print(k)` debug lines.

The local upstream checkout at `~/agilex-arm-gravity-compensation` carries
**uncommitted** edits making the same `V189` to `DEFAULT` change, so compare
against `git -C ~/agilex-arm-gravity-compensation show HEAD:<path>`, not against
its working tree.

### The three demos

All stream torque only — `main_gc.py` and `main_jnt_imp.py` at 200 Hz,
`main_tast_imp.py` at 100 Hz:

```python
robot.move_mit(joint_id, 0, 0, 0, 0, tau[joint_id - 1])   # p = v = kp = kd = 0
```

so the joint driver adds nothing of its own and the arm is held entirely by the
torque the Python loop computes. They differ only in what goes on top of the same
dynamics compensation:

| demo | torque law |
|---|---|
| `main_gc.py` | `rnea(q, qd, 0)` — compensation only; the arm can be pushed around freely |
| `main_jnt_imp.py` | `K*(q_des - q) + B*(qd_des - qd) + nle(q, qd)` |
| `main_tast_imp.py` | `w * (J.T @ (Kc*x_err + Bc*(-J @ qd))) + nle(q, qd)` |

In the Cartesian law `x_err` is `[p_des - p, log3(R_des @ R.T)]`, `J` is the
`link6` Jacobian in `LOCAL_WORLD_ALIGNED`, and the damping acts on the flange
velocity itself — `compute_cartesian_torque` takes **no desired velocity**, so a
moving target is tracked by the spring alone and lags.

Validated gains — joint space, `main_jnt_imp.py`:

| quantity | value |
|---|---|
| `k` | `[10, 10, 10, 2, 1, 1]` N·m/rad |
| `b` | `[0.5, 0.8, 0.8, 0.2, 0.2, 0.2]` N·m·s/rad |

task space — these are `CartesianImpedanceController`'s own constructor defaults,
and what `play_joint_angles.py` sets:

| quantity | value |
|---|---|
| `k` | `[200, 200, 200, 5, 5, 5]` — N/m for x,y,z; N·m/rad for rx,ry,rz |
| `b` | `[5, 5, 5, 0.2, 0.2, 0.2]` |
| `joint_torque_weights` | `[1, 1, 1, 0.5, 1, 0.5]` — trims joints 4 and 6 |
| `ee_frame_name` | `link6` |

**The local `main_tast_imp.py` no longer uses that set** — it lowers `k` on y and
z to 100 and the joint 4–6 weights to 0.5, as listed in the deviations above. The
ergodic run uses a third set again (§3, `run_ergodic_pipeline.make_controller`).
So read the gains from the file you are about to run, not from this table.

Every demo holds the pose it starts in. `R_world_base` is the identity (base
mounted upright).

**The exit pattern matters.** On `KeyboardInterrupt` or any other exception, the
demos switch every joint to a position hold at its current angle:

```python
robot.move_mit(joint_id, joint_angles[joint_id - 1], 0, 10, 0.8, 0)
```

That hands the arm to the joint driver's own PD loop (`kp=10`, `kd=0.8`), so it
stays up after Python exits. Every control loop must end this way. A loop that
simply stops streaming leaves the **last torque latched** (§8): for gravity
compensation that roughly holds the crash pose, but it no longer tracks.

Known rough edges, left alone deliberately: the `tast` typo, bare `except:` in
the exit handlers, Chinese comments, the redundant `sys.path` insertion at the top
of each `main_*.py`.

**Not in `agx_reference`:** IK, trajectory generation, logging, visualisation,
torque or joint-limit bounding, and any moving target. All of those live beside
it (§5) where they exist at all.

**IK is `kinematics/kinematic_solver.py`** — closed form by Pieper decoupling,
which gives up to eight exact branches instead of the one basin a seeded Newton
iteration finds, over the `peg_tcp` frame. It is hardware-free and used offline
only: the online loop needs no IK, because the ergodic controller's output is a
pose the impedance law tracks directly. Its FK and Jacobian come from the same
`AgxPinocchio` wrapper the controllers use, and `test_kinematic_solver.py` checks
the Jacobian against a finite difference of that FK and round-trips IK over
FK-generated poses, including at and near the `q5 = 0` wrist singularity. Use it
rather than writing another solver.

---

## 5. Our code beside it

```
workspace/src/
├── agx_reference/                          # §4; ours there is feed_forward.py
├── execution/executor_helpers.py           # pyAgxArm I/O          (TOUCHES THE ARM)
├── direct_teaching/                        # hardware-free
│   ├── recorder/joint_angle_recorder.py    # JointAngleRecorder, load_recording
│   ├── player/joint_angle_player.py        # JointAnglePlayer: recording -> q(t)
│   ├── player/tracking_error.py            # flange tracking error of a replay, and its figure
│   └── distribution/pose_distribution.py   # PoseDistribution: time-weighted GMM in the cube
├── ergodic_controller/                     # hardware-free
│   ├── ergodic_controller.py               # ErgodicController: cube state -> next cube state
│   └── Ergodic_Exploration_using_TT_PiPER.ipynb   # the E2T2 notebook this was ported from
├── kinematics/kinematic_solver.py          # hardware-free: FK, Jacobian, closed-form IK (§4)
├── visualization/visualizer.py             # hardware-free: offline views, and LiveView
├── simulation/                             # hardware-free
│   ├── meshcat_scene.py                    # MeshCat drawing in the world frame
│   └── fake_executor_helpers.py            # a fake arm, so an execution script runs offline
├── record_joint_angles.py                  # TOUCHES THE ARM: gravity comp + 100 Hz recording
├── play_joint_angles.py                    # TOUCHES THE ARM: replay under Cartesian impedance
├── identify_friction.py                    # TOUCHES THE ARM: friction from sweeps
├── make_sweep_recording.py                 # hardware-free: a synthetic recording to replay
└── run_ergodic_pipeline.py                 # TOUCHES THE ARM: the online peg-in-hole run (§3)
```

There are no `__init__.py` files; these are namespace packages. Every `test_*.py`
sits beside the module it tests.

### `execution/executor_helpers.py`

The SDK boundary shared by the two scripts, so each script is a thin copy of an
upstream loop:

| function | does |
|---|---|
| `connect_arm()` | `PiperFW.DEFAULT` on `can0`, `connect`, loop on `enable()`, wait until `get_joint_angles()` is not `None` |
| `read_joint_velocities(robot)` | (6,) rad/s via six `get_motor_states(i)` calls |
| `apply_joint_torques(robot, tau)` | raises `RuntimeError` if any `|tau|` exceeds `8*b*c`, else `move_mit(j, 0, 0, 0, 0, tau[j-1])`; on an SDK exception it prints and carries on, as upstream does |
| `hold_current_pose(robot, q)` | the §4 exit hold; tries every joint even if one fails |

**Nothing is clipped, but an over-limit torque is refused.** Torques otherwise go
out exactly as `agx_reference` sends them. What is new is the pre-check: the SDK
would silently clamp to `±8*b*c` and print a warning (§8), and a clamped joint has
lost its damping and oscillates at the limit, so `apply_joint_torques` raises
instead and lets the caller's `finally` hold the arm. A run that ends this way is
a gain problem, not a glitch.

### Recording — `record_joint_angles.py`

The `main_gc.py` loop (100 Hz here, `rnea(q, qd, 0)`), plus
`JointAngleRecorder.sample(t_now, q)` fed the `q` the loop already reads. The
operator backdrives the arm through the poses that matter. On exit a `finally`
block holds the pose **first**, then writes the file, so a failed write cannot
leave the arm unheld — and the file is written on any exit, not only Ctrl-C.

`JointAngleRecorder` samples on a fixed grid of `k / rec_freq_hz` from the first
call. After an overrun it jumps past every elapsed slot, so an overrun drops
samples rather than bunching them: the gaps in `t` are real.

`REC_FREQ_HZ` and `CONTROL_FREQ_HZ` are both 100, so every control cycle is
recorded; the recorder's grid still matters, because it is what makes an overrun
drop a sample rather than shift the timeline.

**Format:** `workspace/output/joint_angles_<YYYYmmdd_HHMMSS>.npz` with `t` (N,) s
from the first sample (`t[0] = 0`) and `q` (N, 6) rad. Read it with
`direct_teaching.recorder.joint_angle_recorder.load_recording(path)`, which
returns `(t, q)` — **trimmed, not raw**: it applies `trim_dwell`, dropping the
stationary samples at both ends (the operator walking to and from the arm) and
restarting `t` at 0. It raises if the recording never leaves its first pose.

### Replay — `play_joint_angles.py`

Takes the recording as a mandatory positional argument — the arm replays whatever
that file holds, so it never picks one implicitly. It runs the `main_tast_imp.py`
loop at 200 Hz with the Cartesian gains of §4's table (the controller's defaults,
not what the local demo file now sets), except that the target moves: each cycle
it is `FK(link6, player.joint_angles_at(t))`, the full flange pose of the recorded
`q`.

`JointAnglePlayer.load(path, q_start)` prefixes the recording with a joint-space
approach from the current pose, lasting `max(1 s, max|q_rec[0] - q_start| / 0.3
rad/s)`. `joint_angles_at(t)` interpolates each joint linearly and holds the
first or last sample outside the timeline, so after the recording ends the arm
holds its final pose until Ctrl-C, which triggers the exit hold.

Only the flange pose is tracked, not `q`. The impedance law itself takes no
desired velocity (§4), so on its own it would lag a fast demonstration; the replay
makes that up outside the law, adding `FeedForward.compute_torque` with
`qd_des = player.joint_velocities_at(t)` and
`qdd_des = player.joint_accelerations_at(t)` on top of the impedance torque.

### Imports, lint and checks

`agx_reference` is not an installed package and its files import `core.…` and
`controller.…` as top-level packages. So there are **two import roots**,
`workspace/src` and `workspace/src/agx_reference`. `run.sh` puts both on
`PYTHONPATH`, and `pyproject.toml` tells ruff (`src`) and pyright (`extraPaths`)
the same. Code outside `agx_reference` imports `core.…`, `controller.…` and its
own packages at the top of the file, with no `sys.path` manipulation, and runs
from any directory. Import the wrapper as `core.agx_pinocchio`, never
`agx_reference.core.agx_pinocchio`, or the same file loads twice under two names.

**`agx_reference` is excluded from a ruff directory scan, but not from the hook.**
`pyproject.toml` sets `extend-exclude` on that directory and no `force-exclude`,
and ruff applies an exclude to paths it discovers, not to paths given on the
command line. So `ruff check workspace/src` skips it, while
`.claude/hooks/quality-gate.sh` — the PostToolUse hook, which passes the edited
file explicitly and runs `ruff format` then `ruff check`, blocking until clean —
does lint it. That is what holds our `feed_forward.py` to the same standard
(§4), and it is also why an explicit `ruff check` over upstream files floods with
findings that are not ours to fix. Do not use `noqa`: refactor instead. The rule
set includes mccabe complexity 8, `max-branches` 8, `max-statements` 30 and
`max-args` 6, so long functions have to be split.

`ruff check workspace/src` reports 24 findings, every one of them in the tracked
vendor notebook `ergodic_controller/Ergodic_Exploration_using_TT_PiPER.ipynb`.
Our `.py` files are clean; treat that notebook the way `agx_reference` is
treated, and do not tidy it.

**Prove numerical code by a command, not by inspection.** Compare against an
independent computation — an analytic result, a finite difference, a round trip —
never against a value just printed and pasted in (the `numeric-check` skill).

**There is no pytest in the container**, and `pyproject.toml` still points
`testpaths` at a `tests/` directory that does not exist, so the `test_*.py` files
beside each module are run by importing them and calling their `test_*` functions
(41 of them pass as of this writing). A quick offline check needing no arm: both
controllers reduce to gravity compensation at zero error, which at the zero pose
is `[0, 3.188, -2.807, -0.011, -0.235, 0]` N·m, with the `link6` origin at
`[0.0561, 0, 0.2132]` m and `peg_tcp` at `[0.1159, 0, 0.2184]` m — the peg link
is massless, so adding it left the gravity torque unchanged.

### Claude Code configuration (`.claude/`)

- `settings.json` — the PostToolUse lint hook, allowed read-only commands, and a
  deny list of arm-touching commands. **The deny list is badly out of date and
  cannot be relied on.** It names `piper/main_*` and `agx_reference/*`, plus
  `main.py` and `print_joint_limits.py`, which no longer exist (the `piper_sdk`
  era). It does **not** name any of the four scripts that actually touch the arm
  today: `record_joint_angles.py`, `play_joint_angles.py`, `identify_friction.py`
  and `run_ergodic_pipeline.py`. It also denies `python test_*`, which now blocks
  hardware-free tests instead of anything dangerous. Hard rule 1 is the real
  protection and applies regardless.
- `rules/hardware.md` — extra rules loaded when editing
  `agx_reference/piper/**` or `agx_reference/controller/**`.
- `agents/reviewer.md` — a fresh-context diff reviewer for correctness and scope.
- `skills/` — `numeric-check` (offline checks for numerical code),
  `paper-to-spec` (paper to implementable spec, user-invoked only), and
  `weekly-report`.

**Weekly reports** are LaTeX, IROS-paper style, in `weekly_reports/`, one numbered
directory per week (`1/`, `2/`, `3/` so far). The directory is gitignored; see the
`weekly-report` skill.

---

## 6. CAN bring-up — the usual cause of "can't talk to the robot"

`can0` is a **gs_usb** USB-CAN adapter and the kernel does **not** bring it up or
set a bitrate. After every host reboot or re-plug it is `state DOWN`,
`can state STOPPED`, with no `bitrate` field. The PiPER always expects
**1 Mbit/s**.

**Fix (the user runs it; needs sudo, host or container):**

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

or `bash double_PiPER/can_activate.sh can0 1000000` (the same, plus discovery and
renaming; a third argument selects a USB bus-info when several adapters exist).

**Diagnosis, in order:**

```bash
ip -details link show can0     # UP, "can state ERROR-ACTIVE", bitrate 1000000
cat /sys/class/net/can0/statistics/rx_packets   # sample twice; must increase
```

- Interface absent: adapter unplugged or `gs_usb` not loaded.
- Bitrate not exactly 1000000: rejected even when UP.
- `can state BUS-OFF`: wiring or bitrate mismatch.
- UP at the right bitrate but `rx_packets` stuck at 0: see §6.1.

### 6.1 CAN troubleshooting

A powered PiPER broadcasts feedback continuously and unprompted, so `rx_packets`
climbs fast on a healthy link. Zero means nothing reaches the adapter, and the
cause is physical.

**`bus-errors` / `error-warn` / `error-pass` / `bus-off` are permanently 0 on
this dongle** and carry no information. It does not support bus-error reporting
(`berr-reporting on` is rejected with `Operation not supported`). Only
`rx_packets`, `tx_packets`, `tx_errors` and `tx_dropped` mean anything. Likewise
`loopback on` is accepted by the driver but may not be honoured by the firmware,
so a failed loopback test proves nothing. `loopback` is sticky — setting the
bitrate does not clear it; pass `loopback off` explicitly and check that
`ip -details link show can0` prints no `<LOOPBACK>`.

**Causes of zero traffic, in order of likelihood:**

1. **A dead USB-CAN adapter** — historically the actual cause on this bench. A
   failed candleLight still enumerates and configures `can0` perfectly while
   passing no frames, mimicking a wiring fault exactly.
2. **Arm unpowered or E-stop engaged** (twist to release).
3. **The green screw-terminal block** carrying CAN-H/CAN-L is loose — the
   vendor's own first suggestion (`double_PiPER/README.MD` §3). Remedy, in
   order: reseat the terminal, re-plug the adapter's USB, **power-cycle the
   arm**, re-activate `can0`, restart the program.
4. CAN cable not seated, H/L swapped, or missing 120 Ω termination.

An open circuit gives exactly zero frames *and* zero errors; a bitrate mismatch
or swapped pair would show bus errors — which this dongle cannot report, so the
distinction is unavailable here.

**Prove the software first** with a virtual interface (no hardware, no risk);
20 frames back means can-utils, SocketCAN, the netns and permissions are fine:

```bash
docker exec piper-ergodic bash -lc '
  sudo ip link add dev vcan0 type vcan; sudo ip link set vcan0 up
  ( timeout 3 candump -t d vcan0 > /tmp/v.txt ) &
  for i in $(seq 1 20); do cansend vcan0 123#DEADBEEF; done
  wait; wc -l < /tmp/v.txt; sudo ip link del vcan0'
```

**`tx_packets=0` with `tx_errors` climbing** is the signature of no other node
on the bus to ACK: the controller retransmits forever and the queue backs up into
`ENOBUFS` / "No buffer space available". `cansend can0 000#` discriminates: an
ACKed frame increments `tx_packets` cleanly. If raw `cansend` also fails there is
nothing to fix in Python.

### 6.2 The adapter

`1d50:606f`, "bytewerk candleLight USB to CAN adapter", USB Full Speed. It
reports as a stock candleLight even though the bench calls it an AgileX
USB-CAN-HL; the gs_usb binding is correct. **It is single-channel**: USB
interface `1.0` is the CAN engine and creates `can0`, `1.1` is DFU, and there is
no `can1`. The stock design's 120 Ω termination is an optional solder jumper,
often left unpopulated, which matters at 1 Mbit/s. With power off, resistance
across CAN-H/CAN-L reads ~60 Ω with both ends terminated, ~120 Ω with one, open
for a broken wire, ~0 Ω for a short.

| serial | status |
|---|---|
| `0047001D5246570520323934` | **working** |
| `004C00404148571420343133` | **DEAD — do not put back in service** |

The dead unit died overnight with the arm left powered and the host PC off. It
enumerated and configured `can0` cleanly but never moved a frame; swapping in the
second unit restored communication immediately. **LED as a field indicator:**
the dead unit showed a constant purple LED, the working one varies green/white.
The signal is constant-versus-varying, not hue.

**Diagnostic shortcut: if CAN worked before and broke across a host power-down
with the arm left on, swap the adapter before measuring anything.** Screw
terminals do not fail spontaneously on an untouched bench.

**Habit: shut the arm down before the PC, or unplug the CAN adapter.** The
adapter is USB-bus-powered, so the exposure is "adapter connected but unpowered
while the bus is driven"; breaking either half removes it. **The mechanism is
inferred, not established** — one death, one night, a correlation, and CAN
transceivers are normally specified to tolerate exactly this. Treat it as cheap
insurance; a second death under the same conditions would be real evidence.

---

## 7. Environment

Everything runs in Docker; host `tsukumo3090ti`, user `jens` (uid/gid 1004).
Image `docker-piper-ergodic` (`build.sh`, from `Dockerfile`), container
`piper-ergodic` (`run.sh`: `--rm -it --privileged --net=host --ipc=host`).
Ubuntu 22.04, Python 3.10, with `python-can` 4.6.1, `pyAgxArm` 1.0.0,
`piper_sdk` 1.0.0, `pin` (Pinocchio) 4.1.0, `meshcat`, `robot_descriptions`,
`ttpy[fast]`, numpy/scipy/matplotlib, JupyterLab, `can-utils`.

`piper_sdk` is still installed (the Dockerfile's sanity check imports it) but
nothing in `workspace/src/` uses it; only one of the two SDKs may hold `can0` at
a time.

Mounts (host to container):

| host | container |
|---|---|
| `~/docker_intern_PiPER_ergodic` | `/home/jens/workspace/docker_intern_PiPER_ergodic` |
| `~/agilex-arm-gravity-compensation` | `/home/jens/workspace/agilex-arm-gravity-compensation` |
| `~/Ergodic_Exploration_using_Tensor_Train` | `/home/jens/workspace/Ergodic_Exploration_using_Tensor_Train` |

So the code is at `/home/jens/workspace/docker_intern_PiPER_ergodic/workspace/src`
inside the container.

Claude usually works from the **host** shell, where `pyAgxArm`, Pinocchio and
can-utils are not installed. Run hardware-free Python and read-only inspection
through the container:

```bash
docker exec piper-ergodic bash -lc 'cd /home/jens/workspace/docker_intern_PiPER_ergodic/workspace/src && python ...'
```

`ruff` is on the host, so the lint hook runs there.

**Visualisation.** `--net=host` makes every viewer reachable from the host
browser: meshcat `:7000`, matplotlib WebAgg `:8988`, JupyterLab `:8888`. meshcat
is the default for anything 3D. Static figures go to `workspace/output/`; the one
writer of them, `direct_teaching/player/tracking_error.py`, selects no backend of
its own. **The viewer is `visualization/visualizer.py`**, drawing
through `simulation/meshcat_scene.py`; it uses WebAgg rather than `Agg`, since its
panels are interactive. A new view is a method or a class there, not a new script
(§5).

**The container is not headless.** `--net=host` shares the host loopback, so a
`DISPLAY` naming a TCP display there (`localhost:600x`, from a forwarded X
session) opens a native window with no mounts and no restart. `run.sh` also
forwards the unix socket and an X cookie; the host display is `:1`, not `:0` (the
stale `/tmp/.X11-unix/X0` has no server). Do not conclude "no GUI" from an empty
`DISPLAY` in a fresh `docker exec`.

---

## 8. `pyAgxArm` API notes

Installed at `/usr/local/lib/python3.10/dist-packages/pyAgxArm/` in the
container. Verify signatures there rather than guessing. The drivers under
`protocols/can_protocol/drivers/piper/` (`default/`, `versions/v183|v188|v189/`)
carry long English docstrings — the best documentation for this arm — but read
the code too: the `move_mit` docstring says out-of-range arguments raise
`ValueError`, and the code does not (below).

Why it replaced `piper_sdk`: **it scales `t_ff` to real N·m per firmware
profile** and range-checks every `move_mit` argument instead of silently
bit-masking it. A Cartesian impedance controller puts its entire output through
`t_ff`, so the scaling is load-bearing.

Setup (`connect_arm()` does exactly this):

```python
cfg = create_agx_arm_config(robot=ArmModel.PIPER,
                            firmeware_version=PiperFW.DEFAULT,  # sic: "firmeware"
                            channel="can0")
robot = AgxArmFactory.create_arm(cfg)
robot.connect()
while not robot.enable():
    time.sleep(1)
```

| call | notes |
|---|---|
| `robot.get_joint_angles()` | `None` until the first frame arrives; angles are `.msg`, a 0-indexed 6-list in rad |
| `robot.get_motor_states(i)` | 1-indexed; `.msg.velocity` in rad/s |
| `robot.joint_nums` | 6 |
| `robot.move_mit(joint_index, p_des, v_des, kp, kd, t_ff)` | 1-indexed; `T_ref = kp*(p_des - p) + kd*(v_des - v) + t_ff` |
| `robot.move_p(pose6)` | Cartesian position move; unused here |

### `move_mit` ranges, clamping and quantisation (`PiperFW.DEFAULT`)

**Only a bad `joint_index` raises.** Every other out-of-range argument is
**clamped, with a `Warning: ...` line printed** — per joint, per call, so at
200 Hz a saturated torque shows up as a flood of those lines, not as an
exception. Encoding truncates, so a step is `span / (2^bits - 1)`:

| arg | clamp range | bits | step |
|---|---|---|---|
| `p_des` | ±12.5 rad (see below) | 16 | 3.81e-4 rad |
| `v_des` | ±45.0 rad/s | 12 | 2.20e-2 rad/s |
| `kp` | 0 … 500 | 12 | 0.122 |
| `kd` | ±5.0 | 12 | 2.44e-3 |
| `t_ff` | ±8·b·c: **±32.0 N·m joints 1–3, ±6.506 N·m joints 4–6** | 8 | **0.251 N·m joints 1–3, 0.051 N·m joints 4–6** |

`b` and `c` are `cfg["joint_torque_b"] = (4,4,4,1,1,1)` and
`cfg["joint_torque_c"] = (1,1,1,0.813252,…)`. The `t_ff` quantisation is coarse:
a quarter of a N·m on the big joints. `kd` capping at 5.0 while `kp` runs to 500
means damping runs out well before stiffness does.

`p_des` is clamped to the joint-limit table only after
`robot.set_joint_limits_enabled(True)`; the default is `False`, nothing here
enables it, so the clamp is ±12.5 rad.

### Firmware profile

`PiperFW` offers `DEFAULT`, `V183`, `V188`, `V189`. `V183` is documented as
firmware v183–v187 (S-V1.8-3 … S-V1.8-7), `V188` as S-V1.8-8, `V189` as
S-V1.8-9 and up. **This arm reports S-V1.8-2, below all of them, so
`PiperFW.DEFAULT` is correct.** Everything in the tree passes it. If the arm is
ever flashed, revisit this.

**The profiles are not interchangeable for torque.** All four carry identical
`joint_torque_k/b/c` constants, but the drivers use them differently:

| profile | `t_ff` limit | joints 1–3 | joints 4–6 |
|---|---|---|---|
| `DEFAULT` | ±8·b·c, sent as `t_ff/(b·c)` | ±32.0 N·m | ±6.506 N·m |
| `V183` | ±8·c, sent as `t_ff/c` | ±8.0 N·m | ±6.506 N·m |
| `V188`, `V189` | ±16·c, sent as `t_ff/c` | ±16.0 N·m | ±13.01 N·m |

So the wrong profile mis-scales every torque the loop sends. Upstream's demos use
`V189`, which is why §4's copy differs from them.

### Joint limits

In the config dict as `cfg["joint_limits"]`; read them from there rather than
retyping. Reproduced because any binning over joint angles needs them as its
support:

| joint | rad | deg | note |
|---|---|---|---|
| 1 | −2.6180 … 2.6180 | ±150 | base yaw |
| 2 | 0.0 … 3.1416 | 0 … 180 | **one-sided**, shoulder |
| 3 | −2.9671 … 0.0 | −170 … 0 | **one-sided**, elbow |
| 4 | −1.7453 … 1.7453 | ±100 | forearm roll |
| 5 | −1.2217 … 1.2217 | ±70 | wrist pitch |
| 6 | −2.0944 … 2.0944 | ±120 | wrist roll |

**Joints 2 and 3 are one-sided, so the all-zeros pose sits exactly on their
limits.** Zero is still a valid pose and does not self-collide; joint 2 can only
travel positive and joint 3 only negative. The spans — 300°, 180°, 170°, 200°,
140°, 240° — are very unequal, so binning or normalising over joint space must be
per joint.

**Torque control has no joint-limit protection.** The firmware's soft limits act
on position setpoints; nothing stops a commanded `t_ff` from driving a joint into
its stop, and with `kp = kd = 0` the driver contributes no restoring force. **No
loop watches `q` against this table** — they mirror `agx_reference`, whose targets
are always reachable poses. A controller whose target can wander (the ergodic one)
is where that has to be decided.

The torque magnitude, unlike `q`, is guarded: everything that goes through
`execution/executor_helpers.apply_joint_torques` refuses a `tau` beyond `±8*b*c`
rather than letting the SDK clamp it silently (§5).

### Firmware behaviour, established by experiment in the `piper_sdk` era

`pyAgxArm` frames the mode word for you, so the trap below should not be
reachable through its API, but the firmware is unchanged. This comes from
experiment, not a vendor document.

**Three independent pieces of state:** per-joint driver enable (**persistent** —
an arm enabled in the last run is still enabled after a restart, and it never
times out); the mode word (CAN `0x151`); and the setpoint, which lives on a
*different CAN ID per mode*.

> **Changing the mode word invalidates the current setpoint. Supply a new one in
> the same frame pair, or do not change the mode at all.**

"Limp" is never about enable. It means the active controller has no valid
setpoint, so it commands zero torque — and with no brakes (§2) the arm falls. A
MIT setpoint `(pos, vel, kp, kd, τ)` and a position setpoint (an angle for the
trapezoidal interpolator) are not interchangeable, so a mode change cannot carry
the old one forward.

**There is no command watchdog.** The arm latches its last setpoint
indefinitely. Streaming at 200 Hz is about control quality, not about holding —
which is why a crashed loop leaves a stale torque applied rather than dropping
the arm.

---

## 9. Operating notes & gotchas

- **Enter any impedance or torque mode at the current pose.** Read `q` just
  before the switch so the error there is near zero; a distant target produces a
  violent jump. `play_joint_angles.py` does this with its approach segment.
- **`kp = kd = t_ff = 0` is a limp arm** that will fall. Never suggest it on a
  mounted arm without support. The demos run `kp = kd = 0` safely only because
  `t_ff` carries the full gravity torque every cycle.
- **Teaching mode blocks everything.** If the arm ignores commands, check it is
  not in teaching mode (physical button).
- **If the drivers report all six joints disabled**, recover by toggling
  teaching mode and back with the physical button, not by power-cycling.
- **1-indexed joints at the SDK boundary.** `move_mit` and `get_motor_states`
  take 1–6; `get_joint_angles().msg` is 0-indexed. Keep the conversion in
  `executor_helpers`.
- **Units are radians and metres** everywhere, in printed output and
  command-line arguments as well as in code.
- **Rotations come in two flavours.** `agx_reference` uses Pinocchio rotation
  matrices internally and `scipy.spatial.transform.Rotation` at its edges
  (`R.from_euler` for the base orientation). Pick one per function and convert at
  the boundary.
- **Loop overruns.** A cycle that exceeds its own period prints
  `warning: control loop overrun` — 5 ms in `play_joint_angles.py` at 200 Hz,
  10 ms in `record_joint_angles.py` and `run_ergodic_pipeline.py` at 100 Hz. The
  first thing to profile is `read_joint_velocities`: six separate
  `get_motor_states()` calls per cycle.
- **Never draw to MeshCat from inside a control loop.** A message is a zmq round
  trip to the server process, and it is only cheap when the calls come back to
  back: measured against the fake arm, one `LiveView` update cost a median 9.6 ms
  when called a control period apart, against 0.86 ms hot. `LiveView` therefore
  only stores the newest state in `update()` and draws from a daemon thread, which
  works because pyzmq drops the GIL while it blocks. Anything else that wants to
  watch a live loop does the same.

---

## 10. Typical workflow

```bash
# host, once per boot / re-plug — the user runs this (§6)
sudo ip link set can0 down && sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 up

sh run.sh                      # start the container (or attach from VS Code)
```

The user — never Claude — runs, inside the container:

```bash
cd /home/jens/workspace/docker_intern_PiPER_ergodic/workspace/src
python record_joint_angles.py            # teach: backdrive the arm, Ctrl-C saves
python play_joint_angles.py <rec>.npz    # replay that recording (path is required)
python run_ergodic_pipeline.py <rec>.npz # the online run; MeshCat on :7000 (§3)
python identify_friction.py              # friction sweeps, for feed_forward.py

cd agx_reference
python piper/main_gc.py                  # pure gravity compensation
python piper/main_jnt_imp.py             # joint impedance hold
python piper/main_tast_imp.py            # Cartesian impedance hold
```

All run until Ctrl-C, which hands the arm to a position hold.

Claude may run anything hardware-free, through the container:

```bash
docker exec piper-ergodic bash -lc 'cd /home/jens/workspace/docker_intern_PiPER_ergodic/workspace/src && python -c "
import numpy as np
from core.agx_pinocchio import AgxPinocchio
p = AgxPinocchio(\"agx_reference/piper/piper/urdf/piper_description.urdf\")
print(p.inverse_dynamics(np.zeros(6), np.zeros(6), np.zeros(6)))"'
```
