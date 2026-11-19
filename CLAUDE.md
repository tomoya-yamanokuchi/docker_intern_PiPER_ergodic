# CLAUDE.md — PiPER Ergodic

Guidance for Claude Code in this repository. It is the authoritative document;
`README.md` is a short human-facing intro and lags behind. The implementation is
`final/`, and `final/CLAUDE.md` covers it: read that before working there. This file
holds what is true of the whole repository: the hard rules, the hardware, CAN, the
SDK and the environment.

---

## 1. Hard rules

1. **Never run Python that can move the robot.** A real AgileX PiPER arm is
   physically connected. In `final/src/` that is `record_master.py`,
   `teach_exploration.py` and `run_trials.py` (`simulate.py` drives them on a fake
   arm and is safe) and, generally, anything that imports `pyAgxArm`, `arm.piper` or
   `arm.control`, or calls `robot.connect()`, `robot.enable()`, `move_mit`, `move_p`
   or `move_j`. The user runs those; Claude writes the code and reads the output the
   user pastes back. Read-only inspection is fine: sources, the installed
   `pyAgxArm`, `ip link show can0`, `candump can0`.

   `robot.py`, `method/`, `visualization/` and `arm/fake.py` import no SDK and are
   safe to run, so a torque law, an FK call, an ergodic step or an analysis of a
   logged run can be checked offline. Keep that split: a new control law or analysis
   goes in a hardware-free module; only a top-level script touches the arm.
   `.claude/hooks/arm-guard.py` enforces this on Bash (see §5).
2. **Never create git commits** unless asked, and never add `Co-Authored-By:
   Claude` or `Claude-Session:` trailers.
3. **`final/src/agx_reference/` is third-party code that stays close to upstream**
   (`kehuanjack/agilex-arm-gravity-compensation`, branch `imp`), plus our
   `controller/feed_forward.py`, the reduced 6-DoF URDF and the meshes. Do not
   reformat it, restyle its Chinese comments, or "fix" its bare `except:` clauses.
   New work goes beside it, importing from it. `feed_forward.py` is ours and is
   linted like the rest.
4. **Only `final/src/` is live code.** `double_PiPER/` is the vendor's ROS1 package,
   kept as reference only (what remains is its docs and CAN scripts). No ROS is used
   anywhere. Do not add to, build or "fix" it. The earlier `workspace/src/` tree was
   removed in commit `98d9cc5`; `git show 98d9cc5^:<path>` recovers anything from it.
   `workspace/saved_recordings/` still holds its recordings.

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
re-plug leaves the interface DOWN with no bitrate, §3).

**The exit pattern matters.** On `KeyboardInterrupt` or any other exception, a
control loop switches every joint to a position hold at its current angle:

```python
robot.move_mit(joint_id, joint_angles[joint_id - 1], 0, 10, 0.8, 0)
```

That hands the arm to the joint driver's own PD loop (`kp=10`, `kd=0.8`), so it
stays up after Python exits (`arm.piper.hold_current_pose`). A loop that simply
stops streaming leaves the **last torque latched** (§4), which roughly holds the
crash pose but no longer tracks.

---

## 3. CAN bring-up — the usual cause of "can't talk to the robot"

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
- UP at the right bitrate but `rx_packets` stuck at 0: see §3.1.

### 3.1 CAN troubleshooting

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

### 3.2 The adapter

`1d50:606f`, "bytewerk candleLight USB to CAN adapter", USB Full Speed. It
reports as a stock candleLight even though the bench calls it an AgileX
USB-CAN-HL; the gs_usb binding is correct. **It is single-channel**: USB
interface `1.0` is the CAN engine and creates one `canN`, `1.1` is DFU. A second
`can1` exists only with a second adapter plugged in, and the names follow plug-in
order. Since 2026-11-12 two adapters drive two arms at once; the arm scripts take
`--can <interface>` (default `can0`). The stock design's 120 Ω termination is an
optional solder jumper, often left unpopulated, which matters at 1 Mbit/s. With
power off, resistance across CAN-H/CAN-L reads ~60 Ω with both ends terminated,
~120 Ω with one, open for a broken wire, ~0 Ω for a short.

| serial | status |
|---|---|
| `0047001D5246570520323934` | **working** |
| `004C00404148571420343133` | **working again** (2026-11-12, user report) |

The second unit once failed overnight, with the arm left powered and the host PC
off. It enumerated and configured `can0` cleanly but never moved a frame, and
swapping in the first unit restored communication immediately. On 2026-11-12 the
user put it back in service on the second arm, and it worked. So the failure was
temporary, not a dead unit, and its cause is unknown. **LED as a field
indicator:** in the failed state it showed a constant purple LED, while a working
unit's LED varies green and white. The signal is constant versus varying, not hue.

**Diagnostic shortcut: if CAN worked before and broke across a host power-down
with the arm left on, swap or re-plug the adapter before measuring anything.**
Screw terminals do not fail spontaneously on an untouched bench.

**Habit: shut the arm down before the PC, or unplug the CAN adapter.** The
adapter is USB-bus-powered, so the exposure is "adapter connected but unpowered
while the bus is driven"; breaking either half removes it. **The mechanism is
inferred, not established**: one failure, one night, a correlation, and the
unit recovered, while CAN transceivers are normally specified to tolerate exactly
this. Treat the habit as cheap insurance. A second failure under the same
conditions would be real evidence.

---

## 4. `pyAgxArm` API notes

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

`final/src/arm/piper.py` is the only code that talks to the SDK. Its
`connect_arm()` reads `--fw` and `--can` from `sys.argv` (removing them, so each
script's positional argv is unchanged) and does:

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
100 Hz a saturated torque shows up as a flood of those lines, not as an
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
S-V1.8-9 and up. **The first arm reports S-V1.8-2, below all of them, so
`PiperFW.DEFAULT` is correct for it**, and it is what `arm/piper.py` passes unless
`--fw` says otherwise. The second arm reports S-V1.8-6 and needs `--fw V183`. If an
arm is ever flashed, revisit this.

**The profiles are not interchangeable for torque.** All four carry identical
`joint_torque_k/b/c` constants, but the drivers use them differently:

| profile | `t_ff` limit | joints 1–3 | joints 4–6 |
|---|---|---|---|
| `DEFAULT` | ±8·b·c, sent as `t_ff/(b·c)` | ±32.0 N·m | ±6.506 N·m |
| `V183` | ±8·c, sent as `t_ff/c` | ±8.0 N·m | ±6.506 N·m |
| `V188`, `V189` | ±16·c, sent as `t_ff/c` | ±16.0 N·m | ±13.01 N·m |

So the wrong profile mis-scales every torque the loop sends. Upstream's demos use
`V189`, which is why our copy of them differs.

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
its stop, and with `kp = kd = 0` the driver contributes no restoring force. The
torque magnitude, unlike `q`, is guarded: `arm/piper.apply_joint_torques` refuses a
`tau` beyond the limit above rather than letting the SDK clamp it silently, since a
clamped joint has lost its damping and oscillates at the limit. A run that ends
that way is a gain problem, not a glitch.

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
indefinitely. Streaming at 100 Hz is about control quality, not about holding —
which is why a crashed loop leaves a stale torque applied rather than dropping
the arm.

---

## 5. Environment

Everything runs in Docker; host `tsukumo3090ti`, user `jens` (uid/gid 1004).
Image `docker-piper-ergodic` (`build.sh`, from `Dockerfile`), container
`piper-ergodic` (`run.sh`: `--rm -it --privileged --net=host --ipc=host`).
Ubuntu 22.04, Python 3.10, with `python-can` 4.6.1, `pyAgxArm` 1.0.0,
`piper_sdk` 1.0.0, `pin` (Pinocchio) 4.1.0, `meshcat`, `robot_descriptions`,
`ttpy[fast]`, numpy/scipy/matplotlib, JupyterLab, `can-utils`.

`piper_sdk` is still installed (the Dockerfile's sanity check imports it) but
nothing in `final/src/` uses it; only one of the two SDKs may hold `can0` at a time.

Mounts (host to container):

| host | container |
|---|---|
| `~/docker_intern_PiPER_ergodic` | `/home/jens/workspace/docker_intern_PiPER_ergodic` |
| `~/agilex-arm-gravity-compensation` | `/home/jens/workspace/agilex-arm-gravity-compensation` |
| `~/Ergodic_Exploration_using_Tensor_Train` | `/home/jens/workspace/Ergodic_Exploration_using_Tensor_Train` |

So the code is at `/home/jens/workspace/docker_intern_PiPER_ergodic/final/src`
inside the container. The container's `PYTHONPATH` (set in `run.sh`) still names the
removed `workspace/src`; `final/src/run.sh` overrides it with `final/src` and
`final/src/agx_reference`, which are the two import roots. Always go through it.

Claude usually works from the **host** shell, where `pyAgxArm`, Pinocchio and
can-utils are not installed. Run hardware-free Python and read-only inspection
through the container:

```bash
docker exec piper-ergodic bash -lc 'cd /home/jens/workspace/docker_intern_PiPER_ergodic/final/src && sh run.sh -c "..."'
```

`ruff` is on the host. `final/check.sh` runs it over the tracked Python, then every
`test_*.py` in the container (there is no pytest there). The lint config is
`pyproject.toml`: `final/src/agx_reference` is excluded from a directory scan, but
the edit hook `.claude/hooks/quality-gate.sh` passes the edited file explicitly and
so lints it; that holds our `feed_forward.py` to the standard. Do not use `noqa`:
refactor instead. The rule set includes mccabe complexity 8, `max-branches` 8,
`max-statements` 30 and `max-args` 6.

**Prove numerical code by a command, not by inspection.** Compare against an
independent computation — an analytic result, a finite difference, a round trip —
never against a value just printed and pasted in (the `numeric-check` skill).

**Visualisation.** `--net=host` makes every viewer reachable from the host
browser: meshcat `:7000`, matplotlib WebAgg `:8988`, JupyterLab `:8888`. meshcat is
the default for anything 3D (`final/src/visualization/`). The container is not
headless: a `DISPLAY` naming a TCP display on the shared loopback
(`localhost:600x`, from a forwarded X session) opens a native window, and `run.sh`
also forwards the unix socket and an X cookie; the host display is `:1`, not `:0`.

### Claude Code configuration (`.claude/`)

- `settings.json` — the PostToolUse lint hook, allowed read-only commands, and the
  PreToolUse `hooks/arm-guard.py`: it refuses a Bash command that runs Python whose
  text, or any `.py` file it names, mentions an arm module or script (its regex is
  the list). Every run goes through `docker exec`, which no permission pattern can
  see into. It reads one level only, judges a compound command whole, and so also
  trips on a command that merely names an arm script alongside Python: keep a read
  of an arm file out of a Python run. Hard rule 1 applies regardless.
- `rules/hardware.md` — extra rules loaded when editing `final/src/agx_reference/piper/**`
  or `final/src/agx_reference/controller/**`.
- `agents/reviewer.md` — a fresh-context diff reviewer for correctness and scope.
- `skills/` — `numeric-check`, `paper-to-spec` (user-invoked only), `weekly-report`.

**Weekly reports** are LaTeX, IROS-paper style, in `weekly_reports/`, one numbered
directory per week. The directory is gitignored; see the `weekly-report` skill.

---

## 6. Operating notes & gotchas

- **Enter any impedance or torque mode at the current pose.** Read `q` just
  before the switch so the error there is near zero; a distant target produces a
  violent jump. `run_trials.py` holds the start pose until Enter for this reason.
- **`kp = kd = t_ff = 0` is a limp arm** that will fall. Never suggest it on a
  mounted arm without support. The loops run `kp = kd = 0` safely only because
  `t_ff` carries the full gravity torque every cycle.
- **Teaching mode blocks everything.** If the arm ignores commands, check it is
  not in teaching mode (physical button).
- **If the drivers report all six joints disabled**, recover by toggling
  teaching mode and back with the physical button, not by power-cycling.
- **1-indexed joints at the SDK boundary.** `move_mit` and `get_motor_states`
  take 1–6; `get_joint_angles().msg` is 0-indexed. Keep the conversion in
  `arm/piper.py`.
- **Units are radians and metres** everywhere, in printed output and
  command-line arguments as well as in code.
- **Rotations come in two flavours.** `agx_reference` uses Pinocchio rotation
  matrices internally and `scipy.spatial.transform.Rotation` at its edges. Pick one
  per function and convert at the boundary.
- **Loop overruns** print `warning: control loop overrun`. The first thing to
  profile is `read_joint_velocities`: six separate `get_motor_states()` calls per
  cycle.
- **Never draw to MeshCat from inside a control loop.** A message is a zmq round
  trip to the server process, and it is only cheap when the calls come back to
  back: against the fake arm one update cost a median 9.6 ms when called a control
  period apart, against 0.86 ms hot. The views therefore only store the newest
  state in `update()` and draw from a daemon thread, which works because pyzmq
  drops the GIL while it blocks. Anything else that wants to watch a live loop does
  the same.

---

## 7. Typical workflow

```bash
# host, once per boot / re-plug — the user runs this (§3)
sudo ip link set can0 down && sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 up

sh run.sh                      # start the container (or attach from VS Code)
```

The user — never Claude — runs, inside the container, from `final/src`
(`final/CLAUDE.md` and each script's docstring give the arguments):

```bash
sh run.sh record_master.py       # backdrive the arm through the master trajectory
sh run.sh teach_exploration.py   # confirm exploration datapoints
sh run.sh label.py <master> <exploration>   # hardware-free: the task the law runs on
sh run.sh run_trials.py <e2t2|sliding|retry> <task>.npz [n_trials] [--auto]
```

A second arm adds `--can <interface>` and `--fw V183` to each arm script. All of them
run until Ctrl-C, which hands the arm to a position hold.

---

## Agent skills

### Issue tracker

Issues live in GitHub Issues on tomoya-yamanokuchi/docker_intern_PiPER_ergodic, via `gh`. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default triage labels (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
