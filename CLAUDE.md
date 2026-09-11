# CLAUDE.md — PiPER Ergodic / Cartesian Impedance Control

Guidance for Claude Code when working in this repository. It is the only
document: `PROGRESS.md`, `docs/` and `kinematics/README.md` were all removed in
the cleanup, so anything worth keeping lives here.

---

## 1. Hard rules

1. **Never run Python that can move the robot.** A real AgileX PiPER arm is
   physically connected. Do not execute anything under
   `workspace/src/agx_reference/piper/`, or anything calling `robot.enable()`,
   `robot.connect()`, `move_mit`, `move_p` or `move_j`. The user runs those
   manually; Claude **writes the code and reads the output** the user pastes
   back. Read-only inspection is fine: sources, the installed `pyAgxArm`,
   `ip link show can0`, `candump can0`.

   The hardware line runs through `pyAgxArm`, and it falls inside
   `agx_reference`: **`core/` and `controller/` import no SDK and are safe to
   run**, so a torque law, an FK call or an analysis of recorded data can be
   checked offline; only `piper/main_*.py` open a session with the arm. Keep
   that split when adding code — a new control law belongs in `controller/`,
   where it can be exercised with no arm attached.
2. **Never create git commits**, and never add `Co-Authored-By: Claude` or
   `Claude-Session:` trailers.
3. **`workspace/src/agx_reference/` is third-party code that stays close to
   upstream.** See §4. Do not reformat it, do not restyle its Chinese comments,
   do not "fix" its bare `except:` clauses. `pyproject.toml` excludes it from
   ruff for exactly this reason. **Deliberate extensions are expected** — the
   recorder in §3 is one — but each should be a small, stated diff that still
   reads against upstream, never an incidental rewrite.
4. **Only `workspace/src/` is live code.** `double_PiPER/` is the vendor's ROS1
   package tree, kept purely as reference. No ROS is used anywhere here. Do not
   add to, build or "fix" `double_PiPER/`.

---

## 2. Hardware safety: the arm has no holding brakes

**The PiPER falls the instant it loses motor power.** The vendor documents this
(`double_PiPER/README(EN).md` §2.2): stopping the arm lets it fall with constant
damping, resetting it makes it lose power and fall immediately.

Everything that removes torque drops the arm: cutting AC, the E-stop, disabling
the arm, and MIT mode with `kp = kd = t_ff = 0`.

Before a deliberate power-down: command the arm low and retracted if CAN works,
otherwise physically support it and clear the swing path. Power-cycle sequence:
support the arm → remove AC → wait ~10 s → restore AC → re-activate `can0` (a
USB re-plug leaves the interface DOWN with no bitrate).

---

## 3. What this project is

Direct, ROS-free control of an AgileX PiPER 6-DoF arm from Python over
SocketCAN, working towards an ergodic-exploration controller (E2T2, tensor-train)
driving a peg-in-hole style contact task.

```
E2T2 / ergodic controller  ->  Python 3.10  ->  pyAgxArm
  ->  python-can  ->  socketcan  ->  can0 (gs_usb USB-CAN @ 1 Mbit/s)  ->  PiPER
```

The milestone reached is **Cartesian impedance control**: the end effector
behaves like a spring-damper in task space while the joints stay compliant,
which is what a contact task needs. That is what `agx_reference/` does and it
is validated on this arm.

**The repository was cut back to exactly that.** `workspace/src/` now contains
`agx_reference/` and nothing else — the `piper_sdk` adapter, the hand-written
kinematics, the visualisation application and every doc but this one were
deleted. The rule going forward: build on `agx_reference`, and let the tree grow
again only where the ergodic controller actually needs it. Anything deleted is
recoverable from git history; prefer recovering a specific file over rewriting
it, but do not restore a module speculatively.

### Next milestone: a distribution over joint-angle poses

The task in front of us is to **extend `piper/main_gc.py` to record joint angles
over time and build a distribution of joint-angle poses from them.**

`main_gc.py` is the right host for this because gravity compensation leaves the
arm free to be backdriven by hand: the operator moves it through the poses that
matter and the loop, already running at 200 Hz, is the natural place to sample
`q`. The recorded distribution is the empirical one an ergodic controller needs —
it turns a human demonstration of "where the interesting configurations are"
into the target the E2T2 machinery explores against.

What that implies, and is worth settling before writing code:

- **Record, then analyse.** Writing samples is a loop concern; histogramming or
  fitting them is not, and the analysis must run offline with no arm attached
  (see the hard-rule split in §1). Do not put the estimator inside the 200 Hz
  loop.
- **The loop's timing budget is 5 ms and it already warns when it overruns.**
  Whatever the recorder does per cycle must be cheap — append to a preallocated
  array or a list, and write the file once at exit, not per sample.
- **`main_gc.py` already reads `q` every cycle**, so recording adds no CAN
  traffic; it is a tap on data that is already there.
- **The exit path is where the file gets written**, and it is also what keeps the
  arm up (§4). Do not add a code path that can leave the arm without its
  position-hold hand-off because a file write raised.
- **Joint angles are radians**, and the per-joint support for any binning is the
  limit table in §8 — the ranges are wildly unequal (joint 2 spans 180°, joint 5
  spans 140° but is centred on zero), so a shared bin width across joints would
  be wrong.
- The recording is data, so it belongs in the gitignored `workspace/output/`.

`main_gc.py` is upstream code (§4). Extending it means it stops being
byte-identical, which is fine and expected — but say so, and keep the diff small
enough to read against upstream.

---

## 4. Current state: `agx_reference/` is the working implementation

`workspace/src/agx_reference/` is the **current fully working state of
implementation**, validated on this arm, and since the cleanup it is the only
code in `workspace/src/`. Build the ergodic controller on top of it; do not start
a parallel stack.

It is a copy of `kehuanjack/agilex-arm-gravity-compensation` @ `imp`, kept as
close to verbatim as possible — the point is that it stays the version validated
on this arm, so do not reformat it. Checked against the local upstream checkout
at `~/agilex-arm-gravity-compensation`: `agx_pinocchio.py`,
`task_imp_controller.py`, `main_gc.py` and `main_tast_imp.py` are byte-identical;
`main_jnt_imp.py` and `jnt_imp_controller.py` carry one deliberate change —
`PiperFW.V189` → `PiperFW.DEFAULT`, so all three demos now agree on the profile
this arm's firmware needs — plus stripped trailing whitespace. Six files,
~800 lines:

```
workspace/src/agx_reference/
├── core/agx_pinocchio.py              # Pinocchio wrapper: FK, Jacobian, nle, rnea
├── controller/
│   ├── task_imp_controller.py         # CartesianImpedanceController -> joint torques
│   └── jnt_imp_controller.py          # JointImpedanceController   -> joint torques
└── piper/
    ├── main_gc.py                     # demo 1: pure gravity compensation
    ├── main_tast_imp.py               # demo 2: Cartesian impedance hold [sic: "tast"]
    ├── main_jnt_imp.py                # demo 3: joint impedance hold
    └── piper/{urdf,meshes}/           # its own reduced 6-DoF description
```

All three demos are **torque-only streaming at 200 Hz**:

```python
robot.move_mit(joint_id, 0, 0, 0, 0, tau[joint_id - 1])   # p=v=kp=kd=0
```

so the joint driver adds nothing of its own — the arm is held entirely by the
torque this Python loop computes. The three differ only in what they add on top
of the same `nle` compensation:

| demo | torque law |
|---|---|
| `main_gc.py` | `rnea(q, qd, 0)` — compensation only, arm is free to push around |
| `main_jnt_imp.py` | `K*(q_des - q) + B*(qd_des - qd) + nle` — joint-space spring-damper |
| `main_tast_imp.py` | `w * J.T @ (Kc*x_err + Bc*v_err) + nle` — task-space spring-damper |

Joint impedance holds joint angles; Cartesian impedance holds an end-effector
pose and leaves the null space free, which is the one a contact task wants.

Their validated gains — joint-space, from `main_jnt_imp.py`:

| quantity | value |
|---|---|
| `k` (joint stiffness) | `[10, 10, 10, 2, 1, 1]` N·m/rad |
| `b` (joint damping) | `[0.5, 0.8, 0.8, 0.2, 0.2, 0.2]` |

and task-space, from `main_tast_imp.py`:

| quantity | value |
|---|---|
| `k` (Cartesian stiffness) | `[200, 200, 200, 5, 5, 5]` — N/m for x,y,z, N·m/rad for rx,ry,rz |
| `b` (Cartesian damping) | `[5, 5, 5, 0.2, 0.2, 0.2]` |
| `joint_torque_weights` | `[1, 1, 1, 0.5, 1, 0.5]` — trims the wrist joints |
| `ee_frame_name` | `link6` |
| control rate | 200 Hz |

**The exit pattern matters.** All three demos catch `KeyboardInterrupt` and
switch every joint to a position hold at the angle it is currently at:

```python
robot.move_mit(joint_id, joint_angles[joint_id - 1], 0, 10, 0.8, 0)
```

That hands the arm to the joint driver's own PD loop (`kp=10`, `kd=0.8`) so it
stays up after Python exits. Any new control loop built on these must end the
same way. A loop that simply stops streaming leaves the **last torque latched**
(§8) — for gravity compensation that roughly holds the crash pose, but it no
longer tracks, so the arm sags or drifts if anything disturbs it.

Known rough edges, left alone deliberately: the filename typo
`main_tast_imp.py`; bare `except:` in the exit handlers; Chinese comments. One
worth knowing before extending the loop: `joint_velocities` is gathered through
six separate `get_motor_states()` calls every cycle, which is the first thing to
profile if the 200 Hz loop starts printing its overrun warning.

### What is NOT in `agx_reference`

No IK, no trajectory generation, no logging or telemetry, no visualisation, no
joint-limit enforcement beyond what the firmware does, and no Cartesian target
that moves — the demos hold the pose they start in. Those are the pieces the
ergodic controller will have to add, and the first of them is the recorder in
§3.

**There is no IK in the tree at all any more.** The interface is torque-level, so
none of the demos need one, and the hand-written closed-form solver was deleted
with the rest of `kinematics/`. When the ergodic controller needs to place the
end effector by pose, recover `ik.py` / `ik_closed.py` (plus `fk.py`,
`transform.py`, `model.py` and `selftest.py`) from git history rather than
writing a new one — they were verified against Pinocchio to 2e-16 and round-trip
5000 random poses. Their URDF loader locks gripper joints that
`agx_reference`'s 6-joint URDF does not have, so re-pointing it at that model
takes an edit.

---

## 5. Environment

Everything runs inside Docker; host `tsukumo3090ti`, user `jens` (uid/gid 1004).
Image `docker-piper-ergodic` (`build.sh`), container `piper-ergodic` (`run.sh`,
`--privileged --net=host`), Ubuntu 22.04 / Python 3.10, with `python-can`,
`pyAgxArm`, `piper_sdk` 1.0.0, `ttpy[fast]`, numpy/scipy/matplotlib, `pin` 4.1.0,
`meshcat`.

`piper_sdk` is still installed alongside `pyAgxArm`. Both import fine, but only
one may hold `can0` at a time. Nothing in `workspace/src/` imports `piper_sdk`
any more.

Mounts: `~/docker_intern_PiPER_ergodic` → `/home/jens/workspace/docker_intern_PiPER_ergodic`,
with `~/Ergodic_Exploration_using_Tensor_Train` and
`~/agilex-arm-gravity-compensation` (the upstream of `agx_reference/`) alongside
it. So the code lives at
`/home/jens/workspace/docker_intern_PiPER_ergodic/workspace/src` **inside** the
container.

`--net=host` means `can0` is the same interface on both sides, and every viewer
is reachable from the host browser: meshcat `:7000`, matplotlib WebAgg `:8988`,
JupyterLab `:8888`.

**Visualisation.** meshcat is the default for anything 3D — three.js in the host
browser, so no X display, no OpenGL, no rebuild. Static figures use `Agg` and are
written to the gitignored `workspace/output/`. There is currently **no viewer in
the tree**; the previous one was removed with the `piper_sdk` stack because it
read its poses from that stack's own FK and telemetry. A new one should be built
on `AgxPinocchio.forward_kinematics` and the URDF and meshes already in
`agx_reference/piper/piper/`.

⚠️ **The container is not headless.** `--net=host` shares the host loopback, so a
`DISPLAY` naming a TCP display there (`localhost:600x`, what a forwarded X session
provides) opens a native window with no mounts and no restart. Do not conclude
"no GUI" from a missing `/tmp/.X11-unix` mount or an empty `DISPLAY` in a fresh
`docker exec` — that only rules out the unix-socket route, which `run.sh` also
forwards (display `:1`, not `:0`; the stale `/tmp/.X11-unix/X0` has no server).

Claude usually works from the **host** shell, where `pyAgxArm` is not installed.
To inspect it: `docker exec piper-ergodic bash -lc '<read-only command>'`.
`can-utils` (`candump`, `cansend`) also lives in the container, not on the host.

The empty `workspace/docker_intern_PiPER_egodic` (note the typo) and
`workspace/Ergodic_Exploration_using_Tensor_Train` are leftover mount points.

---

## 6. CAN bring-up — the usual cause of "can't talk to the robot"

`can0` is a **gs_usb** USB-CAN adapter and the kernel does **not** bring it up or
set a bitrate. After every host reboot or re-plug it is `state DOWN`,
`can state STOPPED`, with no `bitrate` field. The PiPER expects **1 Mbit/s**,
always.

**Fix (user runs; needs sudo, host or container):**

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

or `bash double_PiPER/can_activate.sh can0 1000000` (same, plus discovery and
renaming; a third argument selects a USB bus-info when several adapters exist).

**Diagnosis, in order:**

```bash
ip -details link show can0     # UP, "can state ERROR-ACTIVE", bitrate 1000000
cat /sys/class/net/can0/statistics/rx_packets   # sample twice; must increase
```

- Interface absent → adapter unplugged or `gs_usb` not loaded.
- Bitrate not exactly 1000000 → rejected even when UP.
- `can state BUS-OFF` → wiring or bitrate mismatch.
- UP at the right bitrate but `rx_packets` stuck at 0 → see §7.

---

## 7. CAN troubleshooting

A powered PiPER broadcasts feedback continuously and unprompted, so `rx_packets`
should climb fast on a healthy link. Zero means nothing is reaching the adapter,
and the cause is physical.

⚠️ **`bus-errors` / `error-warn` / `error-pass` / `bus-off` are permanently 0 on
this dongle** and carry no information — it does not support bus-error reporting
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
   vendor's own first suggestion (`double_PiPER/README.MD` §3), whose remedy is:
   reseat the terminal → re-plug the adapter's USB → **power-cycle the arm** →
   re-activate `can0` → restart the program, in that order.
4. CAN cable not seated, H/L swapped, or missing 120 Ω termination.

An open circuit gives exactly zero frames *and* zero errors; a bitrate mismatch
or swapped pair would show bus errors — which this dongle cannot report, so the
distinction is unavailable here.

**Prove the software first** with a virtual interface (zero risk, no hardware);
20 frames back means can-utils, SocketCAN, the netns and permissions are fine:

```bash
docker exec piper-ergodic bash -lc '
  sudo ip link add dev vcan0 type vcan; sudo ip link set vcan0 up
  ( timeout 3 candump -t d vcan0 > /tmp/v.txt ) &
  for i in $(seq 1 20); do cansend vcan0 123#DEADBEEF; done
  wait; wc -l < /tmp/v.txt; sudo ip link del vcan0'
```

**`tx_packets=0` with `tx_errors` climbing** is the textbook signature of no
other node on the bus to ACK: the controller retransmits forever, the queue backs
up into `ENOBUFS` / "No buffer space available". `cansend can0 000#` is the
discriminating test — an ACKed frame increments `tx_packets` cleanly. If raw
`cansend` also fails there is nothing to fix in Python.

### The adapter in this setup

`1d50:606f`, "bytewerk candleLight USB to CAN adapter", USB Full Speed. It
reports as a stock candleLight even though the bench calls it an AgileX
USB-CAN-HL; the gs_usb binding is correct. **It is single-channel** — USB
interface `1.0` is the CAN engine and creates `can0`, `1.1` is DFU. There is no
`can1` on this hardware. Note the stock design's 120 Ω termination is an
optional solder jumper often left unpopulated, which matters at 1 Mbit/s; with
power off, resistance across CAN-H/CAN-L reads ~60 Ω both ends terminated,
~120 Ω one end, open = broken wire, ~0 Ω = shorted.

| serial | status |
|---|---|
| `0047001D5246570520323934` | **working** |
| `004C00404148571420343133` | **DEAD — do not put back in service** |

The dead unit died overnight with the arm left powered and the host PC off. It
enumerated perfectly and configured `can0` cleanly but never moved a frame;
swapping in the second unit restored communication immediately, isolating the
fault to that board. **LED as a field indicator:** the dead unit showed a
constant purple LED, the working one varies green/white. The useful signal is
constant-vs-varying, not hue — activity indication modulates with traffic.

**Diagnostic shortcut: if CAN worked previously and broke across a host
power-down with the arm left on, swap the adapter before measuring anything.**
Screw terminals do not fail spontaneously on an untouched bench.

**Habit: shut the arm down before the PC, or unplug the CAN adapter.** The
adapter is USB-bus-powered, so the exposure is "adapter connected but unpowered
while the bus is driven"; breaking either half removes it, and the PC does not
need to stay on. ⚠️ **The mechanism is inferred, not established** — one death,
one night, a correlation. The obvious explanation (current into a dead supply
rail through the transceiver's body diodes) is weaker than it sounds, since CAN
transceivers are normally specified to tolerate exactly this. Treat it as cheap
insurance. A second death under the same conditions would be real evidence.

---

## 8. `pyAgxArm` API notes

Installed at `/usr/local/lib/python3.10/dist-packages/pyAgxArm/`. Verify
signatures there rather than guessing. The per-firmware drivers under
`protocols/can_protocol/drivers/piper/` carry long, accurate English docstrings —
they are the best documentation available for this arm.

Why this SDK replaced `piper_sdk`: **it scales `t_ff` to real N·m per firmware
profile and validates the range**, raising `ValueError` instead of silently
bit-masking an out-of-range value. A Cartesian impedance controller puts its
entire output through `t_ff`, so both properties are load-bearing.

Setup, as both demos do it:

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
| `robot.get_joint_angles()` | returns `None` until the first frame arrives; angles are `.msg`, a 6-list in rad |
| `robot.get_motor_states(i)` | 1-indexed; `.msg.velocity` in rad/s |
| `robot.joint_nums` | 6 |
| `robot.move_mit(joint_index, p_des, v_des, kp, kd, t_ff)` | 1-indexed; `T_ref = kp*(p_des - p) + kd*(v_des - v) + t_ff` |
| `robot.move_p(pose6)` | Cartesian position move; not used by the impedance demos |

`move_mit` ranges and quantisation, from the driver docstring:

| arg | range | step |
|---|---|---|
| `p_des` | ±12.5 rad | 3.81e-4 rad |
| `v_des` | ±45.0 rad/s | 2.20e-2 rad/s |
| `kp` | 0 … 500 | 0.122 |
| `kd` | ±5.0 | 2.44e-3 |
| `t_ff` | ±(8·b·c): **±32.0 N·m joints 1–3, ±6.506 N·m joints 4–6** | — |

The `t_ff` limit comes from `joint_torque_b` and `joint_torque_c` in the config
dict (`[4,4,4,1,1,1]` and `[1,1,1,0.813252,…]`), so it is inspectable at
runtime rather than hardcoded. `kd` capping at 5.0 while `kp` runs to 500 means
damping runs out well before stiffness does.

### Joint limits

Also in the config dict, as `cfg["joint_limits"]` — read them from there rather
than retyping these numbers. Reproduced here because the doc that held them was
deleted, and because any histogram over joint angles needs this as its support:

| joint | rad | deg | note |
|---|---|---|---|
| 1 | −2.6180 … 2.6180 | ±150 | base yaw |
| 2 | 0.0 … 3.1416 | 0 … 180 | **one-sided** — shoulder |
| 3 | −2.9671 … 0.0 | −170 … 0 | **one-sided** — elbow |
| 4 | −1.7453 … 1.7453 | ±100 | forearm roll |
| 5 | −1.2217 … 1.2217 | ±70 | wrist pitch |
| 6 | −2.0944 … 2.0944 | ±120 | wrist roll |

⚠️ **Joints 2 and 3 are one-sided, so the all-zeros pose sits exactly on their
limit.** Zero is still a valid pose and does not self-collide, but joint 2 can
only travel positive and joint 3 only negative. The ranges are also very
unequal — 300°, 180°, 170°, 200°, 140°, 240° — so anything binning or
normalising over joint space must do it per joint.

⚠️ **Torque control has no limit protection.** The firmware's soft limits act on
*position* setpoints; nothing stops a commanded `t_ff` from driving a joint into
its stop, and the demos run `kp = kd = 0` so the joint driver contributes no
restoring force. A controller that generates torques must bound its own output
and watch `q` against this table itself.

**Firmware profile.** `PiperFW` offers `DEFAULT`, `V183`, `V188`, `V189`. The
`V183` driver is documented as firmware v183–v187 = S-V1.8-3 … S-V1.8-7. **This
arm reports S-V1.8-2, which is below that range, so `PiperFW.DEFAULT` is the
correct profile** — checked against the driver docstrings. All three demos pass
`DEFAULT`; `main_jnt_imp.py` arrived from upstream with `V189` and was corrected.
If the arm is ever flashed, revisit this.

For `t_ff` the profile happens not to matter: all four PIPER profiles carry
identical `joint_torque_k/b/c`, so the scaling and the ±32 / ±6.506 N·m limits
are the same (checked at runtime, not assumed). The version drivers do override
other methods, so the profiles are not interchangeable in general — which is why
matching the real firmware is still the right thing to do.

### Firmware behaviour inherited from the `piper_sdk` era

`pyAgxArm` frames the mode word for you, so the trap below should not be
reachable through its API, but the firmware is unchanged and the behaviour is
worth knowing. It was established here by experiment, not from any vendor
document.

**Three independent pieces of state:** per-joint driver enable (**persistent** —
an arm enabled in the last run is still enabled after a restart, and it never
times out); the mode word (CAN `0x151`); and the setpoint, which lives on a
*different CAN ID per mode*.

> **Changing the mode word invalidates the current setpoint. Supply a new one in
> the same frame pair, or do not change the mode at all.**

"Limp" is never about enable. It means the active controller has no valid
setpoint, so the loop commands zero torque — and with no brakes (§2) the arm
falls. A MIT setpoint is `(pos, vel, kp, kd, τ)` and a position setpoint is an
angle for the trapezoidal interpolator; they are not interchangeable, so a mode
change cannot carry the old one forward.

**There is no command watchdog.** The arm latches its last setpoint
indefinitely. Streaming at 200 Hz is about control quality, not about holding —
which is why a crashed loop leaves a stale torque applied rather than dropping
the arm (§4).

---

## 9. Code layout (`workspace/src/`)

```
workspace/src/
└── agx_reference/              # the whole of it (§4)
    ├── core/agx_pinocchio.py          # hardware-free: FK, Jacobian, nle, rnea
    ├── controller/                    # hardware-free: torque laws
    │   ├── task_imp_controller.py
    │   └── jnt_imp_controller.py
    └── piper/                         # TOUCHES THE ARM
        ├── main_gc.py  main_jnt_imp.py  main_tast_imp.py
        └── piper/{urdf,meshes}/       # its own reduced 6-DoF description
```

That is the entire tree. `workspace/output/` is gitignored and is where
generated data and figures go.

**Imports.** `agx_reference` is not an installed package and has no
`__init__.py`. Each `piper/main_*.py` puts its own parent directory on
`sys.path` and then imports `core.…` / `controller.…`, so **they are run from
`workspace/src/agx_reference/` as `python piper/main_gc.py`**. A new module that
wants to import `core` or `controller` must either live beside them and be run
the same way, or repeat that `sys.path` insertion — which is upstream's pattern,
so follow it rather than converting the tree into a package.

⚠️ **Nothing in the tree is linted any more.** `pyproject.toml` excludes
`agx_reference` from ruff (hard rule 3), and it is now the only code, so
`ruff check workspace/src` reports "no Python files found". That protects the
upstream files from being reformatted by the `.claude/hooks/quality-gate.sh`
PostToolUse hook, but it also means new code added inside `agx_reference` gets no
lint or format pass. Worth deciding deliberately when the tree next grows: either
keep new work in a sibling directory that *is* linted, or accept the gap.

**Proving numerical code.** `controller/` and `core/` import no SDK, so any
torque law, FK call or analysis of recorded data can be checked with the arm
unplugged, and should be: correctness is proved by a command, not by inspection.
The deleted `kinematics/selftest.py` is the template worth imitating — it
cross-checked FK and the Jacobian against Pinocchio, and round-tripped 5000
random poses — and it is in git history if you want to read it. For a new
numerical function, compare against an independent computation (an analytic
result, a finite difference, a round trip), never against a value just printed
and pasted in.

## 10. Operating notes & gotchas

- **Enter any impedance or torque mode at the current pose.** Read `q` one
  moment earlier so the position error at the switch is near zero; a distant
  target produces a violent jump.
- **`kp = kd = t_ff = 0` means a limp arm** that will fall. Never suggest it on
  a mounted arm without support. Note the demos run `kp = kd = 0` deliberately
  and are safe only because `t_ff` carries the full gravity torque every cycle.
- **Torque-only control has no joint-limit protection** (§8). Any new
  controller must bound its own output and check `q` itself.
- **Teaching mode blocks everything.** If the arm ignores commands, check it is
  not in teaching mode (physical button).
- **If the drivers report all six joints disabled**, recover by toggling
  teaching mode and back with the physical button, not by power-cycling.
- **1-indexed joints at the SDK boundary.** `move_mit` and `get_motor_states`
  take 1–6, while `get_joint_angles().msg` is a 0-indexed 6-list. The demos
  write `tau[joint_id - 1]`; keep that conversion in one place.
- **Units are radians and metres** everywhere, in printed output and
  command-line arguments as well as in code.
- **Rotations come in two flavours here.** `agx_reference` uses `pin.SE3` /
  Pinocchio rotation matrices internally and `scipy.spatial.transform.Rotation`
  at its edges (`R.from_euler` for the base orientation). Pick one per function
  and convert at the boundary rather than mixing types mid-computation.
- **`get_joint_angles()` returns `None` until the first CAN frame arrives**, so
  every loop waits for it before starting. A recorder must not treat that first
  `None` as a sample.

---

## 11. Typical workflow

```bash
# host, once per boot / re-plug — user runs (see §6)
sudo ip link set can0 down && sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 up

sh run.sh                      # start the container (or attach from VS Code)
cd /home/jens/workspace/docker_intern_PiPER_ergodic/workspace/src
```

Hardware-free, safe for Claude to run — the controllers and the Pinocchio
wrapper import no SDK, so a torque law or an analysis script can be exercised
with the arm unplugged:

```bash
cd workspace/src/agx_reference
python -c "from controller.task_imp_controller import CartesianImpedanceController"
```

Then the user — never Claude — runs one of:

```bash
cd workspace/src/agx_reference
python piper/main_gc.py                        # pure gravity compensation
python piper/main_jnt_imp.py                   # joint impedance hold
python piper/main_tast_imp.py                  # Cartesian impedance hold
```

All three run until Ctrl-C, which hands the arm to a position hold (§4).

A useful sanity check that needs no arm: both controllers must reduce to pure
gravity compensation when the error is zero. At the zero pose that is
`[0, 3.188, -2.807, -0.011, -0.235, 0]` N·m, and the flange sits at
`[0.0561, 0, 0.2132]` m.
