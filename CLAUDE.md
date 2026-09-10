# CLAUDE.md — PiPER Ergodic / MIT Impedance Control

Guidance for Claude Code when working in this repository.

Other docs, not duplicated here: `PROGRESS.md` (what works, known gaps, next
steps), `workspace/src/docs/executables.md` (everything runnable, and how),
`workspace/src/docs/joint-limits.md`, `workspace/src/kinematics/README.md` (how
the IK works).

---

## 1. Hard rules

1. **Never run Python that can move the robot.** A real AgileX PiPER arm is
   physically connected. Do not execute `workspace/src/main.py`, the
   `test_*` scripts, or anything calling `enable_arm`, `move_j`, `ModeCtrl`,
   `JointMitCtrl` or `EmergencyStop`. The user runs those manually; Claude
   **writes the code and reads the output** the user pastes back.
   Read-only inspection is fine: sources, the installed `piper_sdk`,
   `ip link show can0`, `candump can0`. Anything under `kinematics/` or
   `visualization/` is hardware-free and safe to run.
2. **Never create git commits**, and never add `Co-Authored-By: Claude` or
   `Claude-Session:` trailers.
3. **Only `workspace/src/` is live code.** `double_PiPER/` is the vendor's ROS1
   package tree, kept purely as reference. No ROS is used anywhere here. Do not
   add to, build or "fix" `double_PiPER/`.

---

## 2. Hardware safety: the arm has no holding brakes

**The PiPER falls the instant it loses motor power.** The vendor documents this
(`double_PiPER/README(EN).md` §2.2): stopping the arm lets it fall with constant
damping, resetting it makes it lose power and fall immediately.

Everything that removes torque drops the arm: cutting AC, the E-stop,
`disable_arm()`, `EmergencyStop()`, and MIT mode with `Kp = Kd = 0`. This is why
`stop()` waits for a "safe configuration" before `disable_arm()`.

Before a deliberate power-down: command the arm low and retracted if CAN works,
otherwise physically support it and clear the swing path. Power-cycle sequence:
support the arm → remove AC → wait ~10 s → restore AC → re-activate `can0` (a
USB re-plug leaves the interface DOWN with no bitrate).

---

## 3. What this project is

Direct, ROS-free control of an AgileX PiPER 6-DoF arm from Python via
`piper_sdk` over SocketCAN, working towards an ergodic-exploration controller
(E2T2, tensor-train) driving a peg-in-hole style contact task.

The current milestone is **MIT mode (joint impedance) control**: commanding each
joint with `(pos_ref, vel_ref, kp, kd, t_ref)` instead of a position setpoint,
so the arm is compliant.

```
E2T2 / ergodic controller  ->  Python 3.10  ->  piper_sdk (1.0.0, 1_0_0_beta)
  ->  python-can  ->  socketcan  ->  can0 (gs_usb USB-CAN @ 1 Mbit/s)  ->  PiPER
```

---

## 4. Environment

Everything runs inside Docker; host `tsukumo3090ti`, user `jens` (uid/gid 1004).
Image `docker-piper-ergodic` (`build.sh`), container `piper-ergodic` (`run.sh`,
`--privileged --net=host`), Ubuntu 22.04 / Python 3.10, with `python-can`,
`piper_sdk` 1.0.0, `ttpy[fast]`, numpy/scipy/matplotlib, `pin` 4.1.0, `meshcat`.

Mounts: `~/docker_intern_PiPER_ergodic` → `/home/jens/workspace/docker_intern_PiPER_ergodic`,
and `~/Ergodic_Exploration_using_Tensor_Train` alongside it. So the code lives at
`/home/jens/workspace/docker_intern_PiPER_ergodic/workspace/src` **inside** the
container.

`--net=host` means `can0` is the same interface on both sides, and every viewer
is reachable from the host browser: meshcat `:7000`, matplotlib WebAgg `:8988`,
JupyterLab `:8888`, telemetry `:9870`.

**Visualisation.** meshcat is the default for anything 3D — three.js in the host
browser, so no X display, no OpenGL, no rebuild. Static figures use `Agg` and are
written to the gitignored `workspace/output/`.

⚠️ **The container is not headless.** `--net=host` shares the host loopback, so a
`DISPLAY` naming a TCP display there (`localhost:600x`, what a forwarded X session
provides) opens a native window with no mounts and no restart. Do not conclude
"no GUI" from a missing `/tmp/.X11-unix` mount or an empty `DISPLAY` in a fresh
`docker exec` — that only rules out the unix-socket route, which `run.sh` also
forwards (display `:1`, not `:0`; the stale `/tmp/.X11-unix/X0` has no server).

Claude usually works from the **host** shell, where `piper_sdk` is not installed.
To inspect it: `docker exec piper-ergodic bash -lc '<read-only command>'`.
`can-utils` (`candump`, `cansend`) also lives in the container, not on the host.

The empty `workspace/docker_intern_PiPER_egodic` (note the typo) and
`workspace/Ergodic_Exploration_using_Tensor_Train` are leftover mount points.

---

## 5. CAN bring-up — the usual cause of "can't talk to the robot"

`can0` is a **gs_usb** USB-CAN adapter and the kernel does **not** bring it up or
set a bitrate. After every host reboot or re-plug it is `state DOWN`,
`can state STOPPED`, with no `bitrate` field, and `piper.init()` then fails with
a misleading error chain — the real cause is the first line:

```
[ERROR] [PIPER] CAN port can0 is not UP.
ConnectionError [CAN socket 'can0' does not exist.]
```

`Piper.init()` → `C_PiperInterface_V2(judge_flag=True)` → `C_STD_CAN` →
`JudgeCanInfo()`, which raises if the port is missing, not UP, **or** not at
exactly 1 000 000. The PiPER expects **1 Mbit/s**, always.

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
- Bitrate not exactly 1000000 → `JudgeCanInfo()` rejects it even when UP.
- `can state BUS-OFF` → wiring or bitrate mismatch.
- UP at the right bitrate but `rx_packets` stuck at 0 → see §6.

---

## 6. CAN troubleshooting

A powered PiPER broadcasts feedback (`0x2A1`, `0x2A5`–`0x2A6`) continuously and
unprompted, so `rx_packets` should climb fast on a healthy link. Zero means
nothing is reaching the adapter, and the cause is physical.

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
discriminating test — an ACKed frame increments `tx_packets` cleanly.

The SDK shows the same fault at transmit time, not at `init()`:
`SendCanMessage(SEND_MESSAGE_FAILED (100017))`. Per `C_STD_CAN.CAN_STATUS`,
`100016` = sent, `100017` = bus reads healthy but `bus.send()` threw,
`100018` = bus state itself bad. `100017` is a hardware symptom, **not** an SDK
bug: if raw `cansend` also fails there is nothing to fix in Python.

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

## 7. Code layout (`workspace/src/`)

```
workspace/src/
├── main.py                       # entry point: the hardware demo
├── impedance_control/
│   ├── session.py                # connect_arm(): the ONLY module importing piper_sdk
│   ├── piper.py                  # ensure_can_mode / get_q / enable / stop / recovery
│   ├── mit.py                    # set_mit_mode, compose_mit_targets, check_targets, send_mit
│   ├── motion.py                 # MotionProfile, move_to(), mit2can_park(), tracking report
│   └── telemetry.py              # UDP sender for the viewer (:9870)
├── kinematics/                   # hardware-free; see kinematics/README.md
│   ├── transform.py  model.py  fk.py  ik_closed.py  ik.py
│   ├── dynamics.py               # pin.rnea feedforward, make_tau_ff_fn()
│   ├── dh.py                     # published AgileX DH tables, as a check
│   └── selftest.py               # every check; run with the arm unplugged
├── visualization/                # one application: python -m visualization <mode>
│   ├── app.py                    # the CLI: pose | ik | watch | simulate | replay
│   ├── skeleton.py  meshcat_view.py  live_view.py
│   ├── simulate.py               # the same trajectory code, against ...
│   └── simulated_arm.py          # ... a stand-in for the real arm, no SDK
├── docs/                         # executables.md, joint-limits.md
├── print_joint_limits.py         # read-only: firmware limits vs SDK defaults
└── test_*.py                     # scratch hardware scripts, not maintained
```

**`session.py` is the offline/hardware line.** Everything else is handed the
handles and never constructs one, which is what lets the identical control code
run against `SimulatedArm`. If a tool imports `session.py`, it needs a robot.

**Pinocchio is only a parser and a dynamics engine.** `import pinocchio` appears
in exactly three files — `model.py` (URDF parse), `dynamics.py` (`rnea`) and
`selftest.py` (which checks our numpy replacements against the calls they
replaced). Do not reintroduce it elsewhere.

**`mit2can_park()`'s handover target is a hardcoded `JointCtrl(0, ...)`**, so
`q_park` is expected to be the zero pose. It travels there with `move_to()`
first — a one-shot MIT setpoint leaves the whole distance to the position error,
which a soft `kp` never quite closes, so MOVE J inherited the remainder as one
swing. Read §9 before touching it.

**Visualisation is one application**, not a script per purpose. New views become
modes of `python -m visualization`, not new entry points.

Modules import as a plain package relative to `workspace/src`, so **scripts must
be run with `workspace/src` as the working directory**. New modules go in a
subfolder with an `__init__.py`.

The `test_*.py` scripts are scratch/history. They may break when
`impedance_control/` is refactored — do not spend edits keeping them importable.
⚠️ `test_ctrlPiperJoint_mit_can0_multi.py` leaves MIT with a plain
`move_j(current, v)`, which sends byte 3 as `0x00` and leaves the arm **limp**.
Do not copy that exit; see §9.

### `main.py`

Connects, reports the startup state, enters MIT at the current pose, walks a
list of targets with `move_to()`, and always exits through `mit2can_park()` in
its `finally:`. Telemetry is always on — an unread datagram is dropped by the
kernel, so a viewer can be attached or killed mid-run.

`ensure_can_mode()` makes startup deliberately asymmetric:

- **Clean state** (`all(GetArmEnableStatus())` and `ctrl_mode == 1`, i.e. what
  `mit2can_park()` leaves) → sends **nothing**. The first mode frame is
  `send_mit()`'s, which carries `0xAD`, so byte 3 never flips.
- **Anything else** (crash, power-cycle, teaching button) → `reset_from_mit()`
  if the drivers are off, then `teaching2can_mode()`. This path drops the arm,
  deliberately. `arm_status` in the startup print names the fault.

Gains live in one `TUNING` table keyed by joint. The joint driver runs

```
tau = kp * (q - q_meas) + kd * (qdot_ref - qdot_meas) + tau_ff
```

so steady-state sag is `gravity_torque / kp`: raising `kp` stiffens the arm,
whereas putting the gravity-hold torque in `tau_ff` removes the sag while
leaving it compliant — which is what a contact task wants, and is why
`make_tau_ff_fn()` exists. `qdot_ref` stays 0 for a step-and-hold, but a
streamed trajectory must carry each joint's desired velocity or `kd` drags
against the motion and the joint lags by roughly `kd * qdot_des / kp`.

On-wire quantisation, worth knowing while tuning
(`piper_interface_v2.py:3062-3066`): `tau_ff` is 8 bits over ±18 Nm (0.14 Nm
steps), and `kd` is capped at 5.0 while `kp` runs to 500 — damping runs out well
before stiffness does.

---

## 8. `piper_sdk` 1.0.0 API notes

Installed at `/usr/local/lib/python3.10/dist-packages/piper_sdk/`. Verify
signatures there rather than guessing; the API changed between versions and much
of the docstring text is Chinese with an English block below. Vendor demos for
other features are in `piper_sdk/demo/V2/`.

Two layers: `Piper` (`api/piper_api.py`, high level, radians, singleton **per CAN
name** — a second `Piper("can0")` returns the same object with `__init__`
skipped) and `C_PiperInterface_V2` (`interface/piper_interface_v2.py`, raw
frames, returned by `Piper.init()`). **MIT control only exists on the low level.**

Startup order matters, and is in `session.connect_arm()`:
`init_soft_joint_limit_on()` **before** `init()` (which snapshots the flags),
then `connect()` — until that runs, `get_joint_states()` returns all zeros and
merely logs "Read thread not opened".

| Call | Notes |
|---|---|
| `piper.get_joint_states()` | `((j1..j6 rad), time_stamp, Hz)` — take `[0]` |
| `piper.get_gripper_states()` | `((angle, effort), time_stamp, Hz)` |
| `piper.move_j(joints_rad_6tuple, v_int_0_100)` | refuses in Teaching mode; needs all joints enabled |
| `piper.enable_arm()` / `disable_arm()` | `disable_arm()` also sends `EmergencyStop(0x02)` |
| `interface.ModeCtrl(ctrl_mode, move_mode, move_spd_rate_ctrl, is_mit_mode)` | see below |
| `interface.JointMitCtrl(motor_num, pos_ref, vel_ref, kp, kd, t_ref)` | CAN IDs 0x15A–0x15F |
| `interface.GetArmStatus().arm_status.ctrl_mode` | vs `ArmMsgFeedbackStatusEnum.CtrlMode` |

`ModeCtrl` arguments: `ctrl_mode` `0x00` standby / `0x01` CAN control;
`move_mode` `0x00` P, `0x01` J, `0x02` L, `0x03` C (feedback also reports `0x04`
MOVE M, `0x05` CPV); `move_spd_rate_ctrl` 0–100 %; `is_mit_mode` `0x00`
position/velocity, `0xAD` MIT, `0xFF` invalid.

`JointMitCtrl` ranges (SDK docstring): `motor_num` 1–6, `pos_ref` ±12.5 rad,
`vel_ref` ±45.0 (vendor uses 0.0), `kp` 0–500 (vendor 10), `kd` ±5.0 (vendor
0.8), `t_ref` ±18.0 Nm (vendor 0.0). This project runs far gentler than the
vendor reference.

⚠️ `FloatToUint()` (`protocol/piper_protocol_base.py:382`) does **not** clamp —
it scales, truncates and bit-masks, so an out-of-range gain reaches the joint as
a plausible-looking wrong value instead of erroring. That is why
`mit.check_targets()` exists.

`ArmMsgFeedbackStatusEnum.CtrlMode`: `STANDBY=0x00`, `CAN_CTRL=0x01`,
`TEACHING_MODE=0x02`, `ETHERNET=0x03`, `WIFI=0x04`, `REMOTE=0x05`,
`LINKAGE_TEACHING_INPUT=0x06`, `OFFLINE_TRAJECTORY=0x07`. `ArmStatus` carries
`EMERGENCY_STOP`, `TARGET_POS_EXCEEDS_LIMIT`, `JOINT_COMMUNICATION_ERR`,
`JOINT_BRAKE_NOT_RELEASED`, `COLLISION_OCCURRED`.

---

## 9. The 0x151 mode word: why the arm goes limp

The central operating principle of this codebase. Three separate bugs had this
same root cause, and each presented as "the arm went limp".

**Three independent pieces of state**, not to be conflated:

1. **Driver enable** — per joint, `driver_enable_status` in 0x261–0x266, set by
   0x471. **Persistent**: an arm enabled in the last run is still enabled after
   a restart (`double_PiPER/README(EN).md:118-120`). It never times out.
2. **The mode word** — CAN 0x151, six bytes.
3. **The setpoint** — on a *different CAN ID per mode*: 0x155–0x157
   (`JointCtrl`, angles), 0x15A–0x15F (`JointMitCtrl`), 0x152–0x154
   (`EndPoseCtrl`).

**"Limp" is never about enable.** It means the active controller has no valid
setpoint, so the loop commands zero torque — and with no brakes (§2) the arm
falls.

Two bytes matter. Byte 1 `move_mode` selects **where the setpoint comes from**
(MOVE J reads 0x155–0x157, MOVE M reads 0x15A–0x15F). Byte 3 `is_mit_mode`
selects **which low-level controller runs** (`0x00` position-velocity, `0xAD`
MIT). Byte 3 is the dangerous one, and it is **write-only** — 0x2A1 echoes byte
1 as `mode_feed` but nothing echoes byte 3, which is why this was hard to find:
`mode_feed: 1` reports MIT as cleared while the invisible byte does the damage.

> **The rule: changing the mode word invalidates the current setpoint. Supply a
> new one in the same frame pair, or do not change the mode at all.**

A target stored under MOVE M is `(pos, vel, Kp, Kd, τ)`; under MOVE J it is an
angle for the trapezoidal interpolator. They are not interchangeable, so a mode
change cannot carry the old setpoint forward. Corollary: **a bare `ModeCtrl`
with no `JointCtrl` / `JointMitCtrl` behind it is always a bug.**

| sent | byte 1 | byte 3 | target? | result |
|---|---|---|---|---|
| `mit2can_mode()` spam *(deleted)* | 0x04→0x01 | 0xAD→0x00 | **no** | limp |
| single `move_j` out of MIT | 0x04→0x01 | 0xAD→**0x00** | yes | limp |
| `mit2can_park()` | 0x04→0x01 | 0xAD→**0xAD** | yes | **holds** |
| `test_ctrlPiperJoint_can0.py` *(deleted)* | unchanged | unchanged | streamed | **holds** |
| startup `enable()`'s `ModeCtrl` | 0x01→0x01 | 0xAD→**0x00** | **no** | limp **+ wedged** |

Rows 2 and 3 are the controlled experiment, identical but for byte 3. Row 4 is
the control proving there is **no command watchdog**: it exits with no cleanup
and the arm keeps holding, because it never changed the mode word. Row 5 is the
worst case — byte 1 never moved, so nothing *looked* like a mode switch, but the
controller was swapped under an actively holding arm with no target behind it.
That latches a fault, drops all six enable bits, and the arm then ignores
everything until teaching mode is toggled with the physical button.

**Consequences here:** `mit2can_park()` keeps byte 3 at `0xAD` and moves only
byte 1, with `JointCtrl` immediately behind — the vendor's own
`handle_go_zero_service` pattern, whose `is_mit_mode` flag exists precisely so
the caller can say which controller is running. `ensure_can_mode()` sends
nothing in the clean state. So the whole program runs at byte 3 = `0xAD`
throughout: first MIT command → park → next run's startup. Only byte 1 ever
moves, always with a target behind it. `reset_from_mit()` is the vendor's
documented MIT exit but goes through STANDBY, dropping the drivers and needing
two re-enables — **recovery, not shutdown**.

**Documented:** the byte layout; `0x04` MOVE M and `0xAD` as the MIT settings;
that `mode_feed` echoes byte 1 and nothing echoes byte 3; that enable persists
across exit; that the reset path passes through standby. **Inferred from
hardware results, not any vendor document:** the setpoint-invalidation rule, the
byte-1-source / byte-3-controller split, and that a byte-3 swap latches the
fault. That is the simplest model fitting all five observations and it has
predicted two fixes — but it is a model, not a spec. If a future transition
contradicts it, the model is what is wrong.

Untested prediction, if anyone wants to confirm it: flipping byte 3 `0xAD` →
`0x00` **with** a `JointCtrl` behind it, from a non-MIT MOVE J hold. This model
says it still goes limp; a plain "mode frames need targets" rule says it holds.

---

## 10. Operating notes & gotchas

- **Teaching mode blocks everything.** `ctrl_mode` reads `0x02`, `move_j`
  silently refuses, and the arm must go through the `stop()` / re-enable
  handshake in `teaching2can_mode()`.
- **Enter MIT at the current pose.** `send_mit` with a `q0` read one moment
  earlier means near-zero position error at the switch; a distant target
  produces a violent jump.
- **Leave MIT with `mit2can_park()`, never a plain `move_j`** (§9).
- **MIT is a streaming interface.** Real impedance control needs the command
  re-sent at ~100–200 Hz; `move_to()` does this. That is about control quality,
  not holding — the arm latches its last setpoint indefinitely and there is no
  watchdog (§9).
- **`Kp = Kd = 0` means a limp arm** that will fall. Never suggest zero gains on
  a mounted arm without support.
- **1-indexed joints everywhere.** `JointMitCtrl` takes `motor_num` 1–6 and
  `get_q()` returns a dict keyed the same way, so `base[joint - 1]` appears only
  once, inside `get_q()`. Anything reading `piper.get_joint_states()` directly
  is still a 0-indexed tuple.
- **If the drivers report `enable=[False]*6`**, recover by toggling teaching
  mode and back with the physical button, not by power-cycling.
- ⚠️ **`stop()`'s safety condition cannot always be satisfied.** Its wait loop
  came verbatim from the vendor demo:
  1. The wait is bounded by `timeout` (default 10 s); on expiry it warns and
     disables anyway. Not a new hazard — `EmergencyStop(0x01)` at the top has
     already removed torque, so the arm is descending under damping throughout.
  2. The joint-5 test is a one-sided band, not a tolerance: it needs
     `0.2094 < q5 < 0.7854`, so an arm settling near neutral or negative on
     joint 5 can never pass. The joint-2 and joint-3 tests do behave like
     tolerances.
  3. **Joint 4 is never checked; joint 5 is checked twice** — a transcription
     slip in the vendor demo, deliberately preserved. Flag it before changing it.

  `stop()` is only reached from the recovery path, so a normal run never hits it.
- `piper_sdk` writes logs into its own installed package directory; the
  Dockerfile chowns `piper_sdk/log` so this works non-root.

---

## 11. Typical workflow

```bash
# host, once per boot / re-plug — user runs (see §5)
sudo ip link set can0 down && sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 up

sh run.sh                      # start the container (or attach from VS Code)
cd /home/jens/workspace/docker_intern_PiPER_ergodic/workspace/src
```

Hardware-free, and to be run before any hardware run:

```bash
python -m kinematics.selftest                          # every model check
python -m visualization watch                          # shell 1: viewer + telemetry
python -m visualization simulate --target ready        # shell 2: the motion, no arm
```

Then the user — never Claude — runs `python main.py`.
`workspace/src/docs/executables.md` has the full set, both pipelines and every
option.
