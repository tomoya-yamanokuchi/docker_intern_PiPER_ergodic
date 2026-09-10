# What can be run

Everything runs **inside the container**, from `workspace/src`:

```bash
cd /home/jens/workspace/docker_intern_PiPER_ergodic/workspace/src
```

That working directory is not optional. Modules import as a plain package
relative to `workspace/src` (`from impedance_control.mit import send_mit`), so
running from anywhere else fails at the first import.

There are two ways to run a trajectory, and they differ only in what sits at the
end of the command stream — a simulated arm, or the real one. Everything else
(the trajectory, the gains, the limit checks, the telemetry, the viewer) is the
same code in both.

---

## Pipeline 1 — Simulation only

No arm, no CAN, no `piper_sdk`. Use this to see what a motion does before it is
ever sent to hardware.

**Two shells. Start the viewer first** — telemetry is UDP, so anything sent
before `watch` is listening is dropped by the kernel and simply never drawn.

```bash
# shell 1 — the viewer
python -m visualization watch
#   opens meshcat, prints a URL, listens on 127.0.0.1:9870

# shell 2 — the motion: travel from one pose to another
python -m visualization simulate --start zero --target ready
```

Open the printed URL in the host browser and the arm moves through the
trajectory. Add curves next to it with `watch --plot traces` if you have a
`DISPLAY`.

`simulate` also works on its own, without a viewer: it still runs the whole
control path and prints the tracking report, so an unreachable IK target or an
out-of-limit angle raises exactly as it would on hardware.

**What is actually being tested.** `simulate` is not a re-implementation. It
runs the real path — `move_to()`'s smoothstep trajectory, `compose_mit_targets()`
attaching the per-joint gains, `check_targets()` enforcing the joint limits, and
the same telemetry — against `SimulatedArm` in place of the CAN transport.

⚠️ **Tracking is perfect: the commanded angle *is* the measured one.** So this
verifies geometry — reachability, joint limits, and the path swept between
waypoints, which is what you watch to judge whether the arm would hit anything.
It says nothing about impedance behaviour: there is no gravity sag, no lag
behind a fast trajectory and no contact, so gains that are far too soft look
perfect here.

**Choosing the trajectory.** `--start` and `--target` each take either a named
pose — `zero` or `ready` — or six comma-separated joint angles in **radians**,
j1 first:

```bash
python -m visualization simulate --start zero --target ready
python -m visualization simulate --start ready --target 0.5,1.2,-0.8,0.3,0.6,-0.4
python -m visualization simulate --start=zero --target=-1.0,1.2,-0.8,0,0,0
```

Defaults are `--start zero --target ready`.

⚠️ **Use the `--target=...` form when the first angle is negative.** Written as
`--target -1.0,...` argparse reads the leading minus as another option and fails
with `expected one argument`. The `=` form has no such problem.

Both poses are checked against the joint limits before anything runs, so a typo
is reported immediately and names the offending joint:

```
ValueError: pose '0,-0.5,0,0,0,0' is outside the joint limits
            -- j2: -0.5000 outside [0.0000, 3.1400]
```

Remember j2 and j3 are one-sided — see [joint-limits.md](joint-limits.md).

`--max-speed RAD_S` and `--rate HZ` set the travel speed of the fastest joint
and the command rate, matching `MAX_SPEED` and `RATE` in `main.py`. They leave
the *path* unchanged, but not the torque it demands: acceleration scales with
the square of the speed, and the feedforward torque with it.

So `--max-speed` is itself worth simulating. The same move that is fine at 3.0
rad/s is rejected at 4.0:

```
ValueError: MIT target out of range
            -- joint 2 tau_ff: -22.5570 outside [-18.0000, 18.0000]
```

Joint 2 carries the whole arm, and ±18 Nm is all an MIT frame can express (see
[joint-limits.md](joint-limits.md)). Finding that here costs nothing; finding it
on the arm means a move that silently under-torques.

---

## Pipeline 2 — Hardware

⚠️ **This moves a real arm.** Read §2 and §9 of `CLAUDE.md` first. The PiPER has
no holding brakes and falls the instant it loses motor torque.

Same shape as pipeline 1, with `main.py` as the sender instead of `simulate`.

```bash
# once per boot or adapter re-plug, on the host or in the container
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip -details link show can0        # UP, ERROR-ACTIVE, bitrate 1000000

# shell 1 — the viewer (optional, and safe to start, stop or restart mid-run)
python -m visualization watch --tcp 0.06

# shell 2 — the run
python main.py
```

**Before you start it:** the swing path must be clear, the arm powered and the
E-stop released. Run the motion through pipeline 1 first and watch it.

**Give `watch` the same `--tcp`** as `main.py`'s `TCP_OFFSET`, or the drawn tool
point is in the wrong place.

**What `main.py` does.** Connects over CAN, prints the startup state, enters MIT
(impedance) mode *at the pose the arm is already holding* so there is no jump at
the switch, then travels: zero → a joint-space target → zero → an IK solution →
zero, holding and reporting tracking error at each. It always exits through
`mit2can_park()`, on Ctrl+C and on an exception alike.

**What it leaves behind.** A clean exit parks at the zero pose and hands over to
CAN MOVE J with the drivers still on, so the arm *holds* that pose after the
process exits — it does not go limp. That state is also what the next run
expects: finding it, `main.py` sends nothing at startup, so the 0x151 mode byte
never flips.

**Tuning knobs** are module constants at the top of the file: `TUNING` (per-joint
`kp`, `kd`, `qdot_ref`, `tau_ff`), `TAU_FF_SCALE`, `MAX_SPEED`, `RATE`,
`TCP_OFFSET`, `HOLD_TIME`. Joint limits and MIT command ranges are in
[joint-limits.md](joint-limits.md).

Telemetry costs one non-blocking `sendto` per step and is always on, so a viewer
can be attached or killed mid-run without touching the arm.

---

## The visualisation application

One tool, five modes: `python -m visualization <mode>`. None of them touch
hardware — the whole application is free of `piper_sdk`, deliberately, so it
cannot command the arm even by accident.

The first two draw the *model*, standing still. The last three show something
*moving*.

### `pose` — draw a named configuration

```bash
python -m visualization pose ready
python -m visualization pose zero --dh
python -m visualization pose branches --web
python -m visualization pose ready --meshcat
```

Draws the link skeleton with a frame triad at each joint, and highlights the
wrist centre — the joint 4, 5 and 6 axes should all pass through that one point,
which is the spherical-wrist property the closed-form solver depends on.

Takes `zero`, `ready` or `branches` (default `ready`). **`branches` is the
interesting one:** it draws every closed-form IK solution for a single pose, each
in its own colour, so you can see the eight elbow/wrist configurations that reach
the same target.

`--dh` overlays the flange position computed from both published AgileX DH
tables onto the URDF model. The markers should sit on the end of the skeleton;
they disagree by about 0.08 mm, far too small to see, and that is the point — an
error large enough to matter would be obvious.

**Reach for it when** you have changed something in `kinematics/` and want to see
whether the arm still looks like an arm.

### `ik` — solve a TCP pose and draw the result

```bash
python -m visualization ik 0.30,0.05,0.15,0,3.1416,0
python -m visualization ik 0.25,0,0.08,0,3.1416,0 --tcp 0.1358 --web
```

One argument, `X,Y,Z,ROLL,PITCH,YAW`, in **metres and radians**. Solves for that
tool pose, prints the joint angles, and draws the solution with the requested
frame as a long triad — if the solve is right, that triad hides underneath the
arm's own end-effector triad.

If the pose is unreachable it says so and draws the target alone, which is
usually enough to explain why.

**Reach for it when** you want to know whether a Cartesian target is reachable
and what posture it needs, before writing it into a trajectory.

### Shared options for `pose` and `ik`

| flag | effect |
|---|---|
| `--tcp M` | TCP offset along the flange z axis [m]. `0.1358` puts it at the gripper finger base |
| `--save PNG` | filename, written under `workspace/output/` unless it carries a directory |
| `--meshcat` | the real STL meshes in the host browser; prints its URL, needs no X display |
| `--web` | interactive matplotlib on port 8988 |

With neither `--meshcat` nor `--web`, a PNG is written and the path printed.
Saving is the default on purpose: a run that renders nothing at all is never
what was wanted.

`--meshcat` is the one to use when the question is whether the model looks like
the real arm, since it draws the actual meshes rather than a line skeleton.

### `watch` — follow whatever is streaming

```bash
python -m visualization watch
python -m visualization watch --plot traces
python -m visualization watch --plot values
python -m visualization watch --plot both
```

Listens on UDP `127.0.0.1:9870` and draws what arrives, from either pipeline. It
sends nothing and commands nothing, so starting, killing or restarting it during
a run changes nothing about that run.

The 3D arm needs no display — meshcat renders in the host browser. `--plot` adds
a native matplotlib window and does need one:

| `--plot` | window |
|---|---|
| `none` (default) | 3D view only |
| `traces` | rolling commanded-vs-measured curves, one row per joint, on fixed axes spanning each joint's real travel |
| `values` | numbers only: commanded, measured and error per joint, worst last |
| `both` | one window each |

Without `DISPLAY` set to a display this container can reach, it says so and
carries on with the 3D view.

**Reach for it when** anything is running — it is the same viewer for both
pipelines.

### `simulate` — travel between two poses with no arm

```bash
python -m visualization simulate --start zero --target ready
```

Places the simulated arm at `--start`, travels to `--target`, and streams every
commanded step to `watch`. Covered in full in
[Pipeline 1](#pipeline-1--simulation-only) above, including the pose formats and
the limit checking.

**Reach for it when** you have a move in mind and want to see it before the arm
does it.

### `replay` — check the viewer itself

```bash
python -m visualization replay
```

Sends a 20-second synthetic sweep instead of a real trajectory, with the
measured angles lagging the commanded ones slightly so the two traces are
distinguishable.

**Reach for it when** you want to know whether the *viewer* works — a blank
`watch` window is otherwise ambiguous between "nothing is sending" and
"something is broken". It tests the receiving end and nothing else.

---

## `python -m kinematics.selftest`

Acceptance checks for the kinematics, run entirely on the model. It never opens
a CAN socket and never imports `piper_sdk` — `check_no_hardware()` enforces that
rather than trusting it — so the arm can be unplugged or unpowered.

Thirteen checks, about six seconds. Among them: forward kinematics and the
Jacobian against finite differences, the static torque against `d(PE)/dq`, both
published DH tables cross-checked against the URDF, all 2400 closed-form IK
branches, and a 5000-sample IK round trip.

Run it after touching anything under `kinematics/`, and before any hardware run.
Every line starts `PASS:` and it ends with `All checks passed.`; anything else is
a real failure.

---

## Ports

All opened inside the container and used from the host browser, which works
because the container runs with `--net=host`.

| port | what |
|---|---|
| 7000 | meshcat (`--meshcat`, and `watch`) — the **next free** port from 7000, so it may land on 7001+ if another view is already up. The URL is always printed. |
| 8988 | matplotlib WebAgg (`--web`) |
| 9870 | telemetry, UDP (`main.py` / `simulate` → `watch`) |

Not documented here: `print_joint_limits.py`, `test_JointMitControl.py` and
`test_ctrlPiperJoint_mit_can0_multi.py`.
