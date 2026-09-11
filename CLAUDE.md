# CLAUDE.md — PiPER Ergodic / Cartesian Impedance Control

Guidance for Claude Code in this repository. It is the authoritative document;
`README.md` is a short human-facing intro and lags behind (it still describes a
tree holding only `agx_reference/`).

---

## 1. Hard rules

1. **Never run Python that can move the robot.** A real AgileX PiPER arm is
   physically connected. The scripts that open a session with it are
   `agx_reference/piper/main_*.py`, `record_joint_angles.py` and
   `play_joint_angles.py` — and, generally, anything that imports `pyAgxArm` or
   `execution.executor_helpers`, or calls `robot.connect()`, `robot.enable()`,
   `move_mit`, `move_p` or `move_j`. The user runs those; Claude writes the code
   and reads the output the user pastes back. Read-only inspection is fine:
   sources, the installed `pyAgxArm`, `ip link show can0`, `candump can0`.

   **`core/`, `controller/` and `direct_teaching/` import no SDK and are safe to
   run**, so a torque law, an FK call, a recording or an analysis of one can be
   checked offline. Keep that split: a new control law or analysis goes in a
   hardware-free module; only a top-level script touches the arm.
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

**Working and validated on the arm:** Cartesian impedance control — the end
effector behaves as a spring-damper in task space with the null space free —
plus joint impedance and pure gravity compensation. That is `agx_reference/`
(§4).

**Built beside it:** direct teaching (§5). The operator backdrives the arm under
gravity compensation while `q` is recorded at 10 Hz; a recording can be replayed
under the Cartesian impedance law.

**Next milestone: a time-spent distribution over the recordings** — a density
whose value at a place is the time the operator spent there. Dwelling is dense,
passing through is sparse. It is the empirical target distribution the E2T2
machinery explores against. Settled so far:

- **Offline only.** It reads `.npz` recordings with no arm attached and belongs
  in a hardware-free module; nothing of it goes into a control loop.
- **Time, not sample count, is the weight.** At 10 Hz a sample stands for about
  0.1 s, but `t` is stored, so a sample's exact weight is the interval it
  covers, and samples dropped by a loop overrun are accounted for rather than
  silently under-weighted.
- **Joint angles are radians**; the per-joint support for binning is the limit
  table in §8. The ranges are very unequal, so no shared bin width.
- **A recording has dwell at both ends** — at the start pose until the operator
  moves the arm, at the end pose until Ctrl-C. It counts as time spent unless
  something decides otherwise.
- Several recordings may feed one distribution.
- Output goes in the gitignored `workspace/output/`.

**Open, to settle before implementing:** whether "place" is the joint
configuration `q` (6-D), the flange position `FK(q)` (3-D), or the full pose —
and so which space the density lives in.

The E2T2 reference is `~/Ergodic_Exploration_using_Tensor_Train` (two notebooks,
NumPy and JAX), mounted into the container (§7). No E2T2 code is in this tree yet.

---

## 4. `agx_reference/` — the validated implementation

A copy of `kehuanjack/agilex-arm-gravity-compensation`, branch `imp`, checked
against upstream `HEAD` `8e2545c`. Six Python files, ~800 lines, plus its own
reduced 6-DoF URDF and meshes:

```
workspace/src/agx_reference/
├── core/agx_pinocchio.py              # Pinocchio wrapper: FK, Jacobian, nle, rnea
├── controller/
│   ├── task_imp_controller.py         # CartesianImpedanceController -> joint torques
│   └── jnt_imp_controller.py          # JointImpedanceController     -> joint torques
└── piper/
    ├── main_gc.py                     # demo: pure gravity compensation
    ├── main_tast_imp.py               # demo: Cartesian impedance hold  [sic: "tast"]
    ├── main_jnt_imp.py                # demo: joint impedance hold
    └── piper/{urdf,meshes}/           # piper_description.urdf, .dae visual, .stl collision
```

**Deviation from upstream `HEAD`:** `agx_pinocchio.py` and
`task_imp_controller.py` are byte-identical. `jnt_imp_controller.py` differs in
whitespace only. The three `main_*.py` pass `PiperFW.DEFAULT` where upstream
passes `PiperFW.V189` (this arm's firmware needs `DEFAULT`, §8), with the comment
`# this arm reports S-V1.8-2` in `main_gc.py` and `main_tast_imp.py`, and
trailing whitespace stripped. Nothing else.

The local upstream checkout at `~/agilex-arm-gravity-compensation` carries
**uncommitted** edits making the same `V189` to `DEFAULT` change, so compare
against `git -C ~/agilex-arm-gravity-compensation show HEAD:<path>`, not against
its working tree.

### The three demos

All stream torque only, at 200 Hz:

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

task space, `main_tast_imp.py` (also the controller's defaults):

| quantity | value |
|---|---|
| `k` | `[200, 200, 200, 5, 5, 5]` — N/m for x,y,z; N·m/rad for rx,ry,rz |
| `b` | `[5, 5, 5, 0.2, 0.2, 0.2]` |
| `joint_torque_weights` | `[1, 1, 1, 0.5, 1, 0.5]` — trims joints 4 and 6 |
| `ee_frame_name` | `link6` |

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
torque or joint-limit bounding, and any moving target. There is **no IK anywhere
in the tree**. A hand-written closed-form solver (`ik.py`, `ik_closed.py`, with
`fk.py`, `transform.py`, `model.py`, `selftest.py`) was verified against
Pinocchio to 2e-16 and round-tripped 5000 random poses; it is in git history
under `kinematics/`. Recover it rather than writing a new one. Its URDF loader
locks gripper joints that this 6-joint URDF lacks, so re-pointing it takes an
edit.

---

## 5. Our code beside it

```
workspace/src/
├── agx_reference/                          # §4; not linted
├── execution/executor_helpers.py           # pyAgxArm I/O          (TOUCHES THE ARM)
├── direct_teaching/                        # hardware-free
│   ├── recorder/joint_angle_recorder.py    # JointAngleRecorder, load_recording
│   └── player/joint_angle_player.py        # JointAnglePlayer: recording -> q(t)
├── record_joint_angles.py                  # TOUCHES THE ARM: gravity comp + 10 Hz recording
└── play_joint_angles.py                    # TOUCHES THE ARM: replay under Cartesian impedance
```

There are no `__init__.py` files; these are namespace packages.

### `execution/executor_helpers.py`

The SDK boundary shared by the two scripts, so each script is a thin copy of an
upstream loop:

| function | does |
|---|---|
| `connect_arm()` | `PiperFW.DEFAULT` on `can0`, `connect`, loop on `enable()`, wait until `get_joint_angles()` is not `None` |
| `read_joint_velocities(robot)` | (6,) rad/s via six `get_motor_states(i)` calls |
| `apply_joint_torques(robot, tau)` | `move_mit(j, 0, 0, 0, 0, tau[j-1])`; prints and carries on on exception, as upstream does |
| `hold_current_pose(robot, q)` | the §4 exit hold; tries every joint even if one fails |

Torques go out exactly as `agx_reference` sends them, with no clipping of our
own. The only bound is the SDK's clamp to the `t_ff` range (§8).

### Recording — `record_joint_angles.py`

The `main_gc.py` loop (200 Hz, `rnea(q, qd, 0)`), plus
`JointAngleRecorder.sample(t_now, q)` fed the `q` the loop already reads. The
operator backdrives the arm through the poses that matter. On exit a `finally`
block holds the pose **first**, then writes the file, so a failed write cannot
leave the arm unheld — and the file is written on any exit, not only Ctrl-C.

`JointAngleRecorder` samples on a fixed grid of `k / rec_freq_hz` from the first
call. After an overrun it jumps past every elapsed slot, so an overrun drops
samples rather than bunching them: the gaps in `t` are real.

**Format:** `workspace/output/joint_angles_<YYYYmmdd_HHMMSS>.npz` with `t` (N,) s
from the first sample (`t[0] = 0`) and `q` (N, 6) rad. Read it with
`direct_teaching.recorder.joint_angle_recorder.load_recording(path)`, which
returns `(t, q)`.

### Replay — `play_joint_angles.py`

Picks the newest `joint_angles_*.npz` by filename, then runs the `main_tast_imp.py`
loop and gains, except that the target moves: each cycle it is
`FK(link6, player.joint_angles_at(t))`, the full flange pose of the recorded `q`.

`JointAnglePlayer.load(path, q_start)` prefixes the recording with a joint-space
approach from the current pose, lasting `max(1 s, max|q_rec[0] - q_start| / 0.3
rad/s)`. `joint_angles_at(t)` interpolates each joint linearly and holds the
first or last sample outside the timeline, so after the recording ends the arm
holds its final pose until Ctrl-C, which triggers the exit hold.

Only the flange pose is tracked, not `q`, and with no velocity feedforward (§4)
the arm lags a fast demonstration.

### Imports, lint and checks

`agx_reference` is not an installed package and its files import `core.…` and
`controller.…` as top-level packages. So there are **two import roots**,
`workspace/src` and `workspace/src/agx_reference`. `run.sh` puts both on
`PYTHONPATH`, and `pyproject.toml` tells ruff (`src`) and pyright (`extraPaths`)
the same. Code outside `agx_reference` imports `core.…`, `controller.…` and its
own packages at the top of the file, with no `sys.path` manipulation, and runs
from any directory. Import the wrapper as `core.agx_pinocchio`, never
`agx_reference.core.agx_pinocchio`, or the same file loads twice under two names.

**`agx_reference` is excluded from ruff; everything else is linted.** The
`.claude/hooks/quality-gate.sh` PostToolUse hook runs `ruff format` and
`ruff check` on every edited `.py` and blocks until it is clean. Do not use
`noqa`: refactor instead. The rule set (`pyproject.toml`) includes mccabe
complexity 8, `max-branches` 8, `max-statements` 30 and `max-args` 6, so long
functions have to be split. `ruff check workspace/src` is currently clean.

**Prove numerical code by a command, not by inspection.** Compare against an
independent computation — an analytic result, a finite difference, a round trip —
never against a value just printed and pasted in (the `numeric-check` skill).
`pyproject.toml` points pytest at `tests/`, which does not exist yet. A quick
offline check needing no arm: both controllers reduce to gravity compensation at
zero error, which at the zero pose is `[0, 3.188, -2.807, -0.011, -0.235, 0]` N·m,
with the `link6` origin at `[0.0561, 0, 0.2132]` m.

### Claude Code configuration (`.claude/`)

- `settings.json` — the PostToolUse lint hook, allowed read-only commands, and a
  deny list of arm-touching commands. The deny list predates the current tree: it
  names `piper/main_*` and `agx_reference/*`, but not `record_joint_angles.py` or
  `play_joint_angles.py`. Hard rule 1 applies regardless.
- `rules/hardware.md` — extra rules loaded when editing
  `agx_reference/piper/**` or `agx_reference/controller/**`.
- `agents/reviewer.md` — a fresh-context diff reviewer for correctness and scope.
- `skills/` — `numeric-check` (offline checks for numerical code),
  `paper-to-spec` (paper to implementable spec, user-invoked only), and
  `weekly-report`.

**Weekly reports** are LaTeX, IROS-paper style, in `weekly_reports/` (currently
only `template/`). The directory is gitignored; see the `weekly-report` skill.

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
is the default for anything 3D. Static figures use `Agg` and go to
`workspace/output/`. There is **no viewer in the tree**; a new one should be
built on `AgxPinocchio.forward_kinematics` and the URDF and meshes in
`agx_reference/piper/piper/`.

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

**Torque control has no limit protection.** The firmware's soft limits act on
position setpoints; nothing stops a commanded `t_ff` from driving a joint into
its stop, and with `kp = kd = 0` the driver contributes no restoring force. No
loop in the tree bounds its torque or watches `q` against this table — they
mirror `agx_reference`, whose targets are always reachable poses. A controller
whose target can wander (the ergodic one) is where that has to be decided.

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
- **Loop overruns.** A cycle that exceeds 5 ms prints
  `warning: control loop overrun`. The first thing to profile is
  `read_joint_velocities`: six separate `get_motor_states()` calls per cycle.

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
python play_joint_angles.py              # replay the newest recording

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
