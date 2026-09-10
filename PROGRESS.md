# Progress

Status of the PiPER control stack, working towards ergodic exploration and a
peg-in-hole contact task.

| area | status |
|---|---|
| MIT (impedance) joint control | working |
| Forward and inverse kinematics | **finished** |
| Gravity / inverse-dynamics feedforward | working, validated on hardware |
| Interpolated joint-space motion | working |
| Live visualisation (meshcat + telemetry) | working, not yet seen with the arm |
| Self-collision checking | not started |
| Ergodic (E2T2) controller | not started |

---

## Kinematics — finished

`workspace/src/kinematics/`. Hardware-free by construction: the selftest asserts
`piper_sdk` and `can` are never imported, so it runs with the arm unplugged.

**The kinematics are now this project's own numpy code, not Pinocchio calls.**
`transform.py` replaces `pin.SE3` and `pin.rpy` (the SO(3) conversions delegate to
`scipy.spatial.transform.Rotation`; only the SE3 container and `log6` are written
out here), `fk.py` walks the joint placements itself, and the Jacobian is the
analytic geometric one. `import pinocchio` now appears in exactly three files:
`model.py` parses the URDF with it, `dynamics.py` runs `rnea`, and `selftest.py`
checks our replacements against the calls they replaced.

That last check is the point: over 2000 random configurations our forward
kinematics agrees with Pinocchio's to 2.2e-16 m, the joint frames to 7.8e-16, both
Jacobian conventions to 1.8e-15, and `log6` to 3.8e-10 — including at a half turn,
which is the normal case here since a tool pointing down is `rpy(0, pi, 0)`. Every
pre-existing number is unchanged: round trip still 5000/5000 with worst position
error 144.051 um, DH agreement still 0.140 mm.

The cost is speed. A solve went from **0.18 ms to 0.75 ms median**, about 4x, and
it is forward kinematics that dominates — 37 us against Pinocchio's compiled 1 us,
because the chain is thirteen small numpy matrix products with Python call
overhead on each. `scipy`'s `log3` is only about 0.2 ms of it; a hand-rolled
`atan2` version would recover that but loses six digits within a nanoradian of a
half turn, so it was not taken. `__slots__` and dropping the `asarray` calls buy
12% and were not worth the loss of clarity. At 0.75 ms a solve still fits thirteen
times over into a 100 Hz step, and IK is not in the streaming loop anyway.

The IK polish also lost its `pin.Jlog6`: it is now a plain world-frame
`[position error; log3(residual rotation)]` damped least-squares step. Measured on
79 near-singular poses where the closed form misses, the old and new forms repair
76 and 75 of them, with the new one showing the smaller worst error, so the
simpler formulation was kept. `pin.integrate` was dropped for plain `q + v`,
which it provably equalled — every joint here is revolute, so the configuration
space is just R^6.

See `kinematics/README.md` for how the solver works.

The arm turned out to be a textbook anthropomorphic 6R **with a spherical wrist** —
joints 4, 5 and 6 meet at a point to within 88 µm, which is a rounding artefact in
the vendor's CAD export. That contradicted the original design assumption and made
closed-form IK available, so the solver enumerates all eight branches rather than
iterating from a seed.

Verified, all without hardware:

- **round trip 5000/5000 (100%)**, worst position error 144 µm, median 0.75 ms
- Jacobian against finite differences, 2.7e-09
- every replaced Pinocchio call, to 1e-15 or better (see above)
- FK against **both** published AgileX DH tables (standard and modified), 0.140 mm
- gravity torque against the numerical gradient of the potential energy, 2.7e-09 Nm
- the two URDF variants bit-identical in every joint placement

A seeded Newton solver was tried first and abandoned: it solves 43% of reachable
poses from one seed, 90% with twenty random restarts, and plateaus at 93% however
many are added. Enumerating the branches finds the answer every time.

A short Gauss-Newton repair remains for poses near the wrist singularity (j5 ≈ 0),
where the split of a rotation between j4 and j6 is ill-conditioned and the 88 µm
idealisation throws the orientation by several degrees. It fires on 3 poses in
5000.

## Dynamics and feedforward

`kinematics/dynamics.py` — one `pin.rnea` call, so gravity is the zero-velocity
special case and a move also gets its inertial and Coriolis terms.

**Validated on hardware at `TAU_FF_SCALE = 1.0`**, which is what lets the arm hold
a pose at low stiffness. Before it, `kp = 1.0` produced 1 Nm of restoring torque
against gravity loads up to 8.55 Nm and the arm simply sagged.

Two URDFs live in `workspace/robot_description/urdf/`, differing only in the
gripper: both keep every gripper link, joint and frame, so they stay
kinematically indistinguishable, but `piper_description_zero_gripper_mass.urdf`
zeroes the inertia of `gripper_base`, `link7` and `link8` **and drops their
`<visual>`**, so a viewer draws the arm as it is actually built. `<collision>` is
untouched in both, which matters for the self-collision work still to come.

That variant is the default because no gripper is currently fitted — using the
loaded one instead costs up to 3.26 Nm of feedforward error on joint 2, which is
23× the `tau_ff` quantisation step.

## Motion

`impedance_control/motion.py` — `move_to()` streams an interpolated MIT trajectory
at 100 Hz on a smoothstep profile, so velocity starts and ends at zero and
`qdot_ref` carries the real intended velocity instead of zero. The feedforward is
evaluated at every step, never interpolated between endpoints; interpolating it
would be wrong by up to 9.5 Nm.

Reports commanded angles next to measured ones with the per-joint error, which is
the quantity being tuned. `dry_run=True` prints the whole command stream and sends
nothing. With a `telemetry` socket it also streams every step to the live viewer.

## Visualiser

`kinematics/visualize.py` — 3D skeleton plot, PNGs into the gitignored
`workspace/output/`, or `--web` for an interactive plot the host browser reaches at
`localhost:8988`. `--target x,y,z,r,p,y` solves IK for a TCP pose and draws the
result. Metres and radians throughout.

`kinematics/meshcat_view.py` — the same poses drawn with the **real STL meshes**,
in the host browser. meshcat renders with three.js, so the container needs no X
display, no OpenGL and no rebuild, and `--net=host` means the printed URL opens
directly on the host. `visualize.py --meshcat` is the entry point; `--pose
branches` cycles through the closed-form solutions on the one arm.

The meshes are placed by *our own* forward kinematics, which makes the picture a
real check rather than a second rendering of the URDF — the round-trip test
compares FK against FK and cannot catch a frame convention that is wrong but
self-consistent. Verified numerically that each link origin lies inside its own
mesh and that every parent mesh reaches its child's origin (4–41 mm), i.e. the
chain is connected.

**Which links are drawn comes from the URDF**, not from a list in the viewer: a
link with no `<visual>` is not drawn. So the default zero-gripper-mass variant
shows six links and no gripper, while the gripper-mass variant still draws
`gripper_base` (and says it is skipping `link7`/`link8`, whose prismatic origins
this viewer does not yet apply). The `<visual>` origin is read and applied too —
it is identity on every link of this arm, but reading it costs three lines and
removes an assumption that would otherwise fail silently.

**The container is not headless**, contrary to what this file said before: with
`--net=host` it shares the host loopback, so a `DISPLAY` naming a TCP display
there — as a forwarded X session provides — opens a native window with no change
to `run.sh` and no restart. On this host `ss -ltn` shows X channels on
`127.0.0.1:6000`–`6003` while Xorg itself runs `-nolisten tcp`.

`run.sh` now *also* forwards the local unix socket (`DISPLAY`, an `xauth` cookie,
`/tmp/.X11-unix`), which covers the remaining case: a session on the machine's
own screen, where the display is `:1`, not `:0`. meshcat needs none of it.

## Live viewer

`tools/live_viewer.py` — a separate process that watches a run in progress: the
3D arm in meshcat plus two optional matplotlib windows. `--traces` gives rolling
commanded-vs-measured plots, one row per joint sharing an x axis so a wrist error
can be read against what the shoulder was doing at that instant, with the live
error printed on each panel. `--values` gives a numbers-only window — the same
`cmd / act / err` rows `report_tracking()` prints, kept up to date, for when the
shape of the motion is already understood and only the magnitude matters. Both
can be up at once. `--replay` feeds a synthetic sweep so it can be checked with
no arm.

`--geometry +X+Y` places the first window (default: the right half of the screen,
leaving the left for the browser). meshcat draws in a browser and these are native
windows, so they cannot share a frame, and meshcat 0.3.2 has no text geometry to
put the numbers into the 3D scene — placing the window beside the tab is as close
as the two get.

Each panel has **fixed** y limits: zero in the middle, that joint's stop at each
edge (`max(|q_min|, |q_max|)`, so the axis stays symmetric even for the one-sided
j2 and j3, whose real stops are drawn as dashed lines). Autoscaling was actively
misleading here — j4 holds at zero apart from a 4e-06 rad IK residual, and
autoscale magnified that into a full-height curve peaking at "3.8" with the
`1e-6` factor in a corner label, which reads as a command to pi on a joint that
is not moving. Fixed axes also make the panels comparable: j2's 1.5 rad swing
should look bigger than j1's 0.46 rad, and under autoscale it did not.

The x axis counts **samples received**, not the sender's clock, and the window is
a bounded deque holding the last 5 s of travel (500 steps at `main.py`'s 100 Hz —
`STREAM_RATE` in the viewer has to match `RATE` there for that to stay true).
`move_to()` measures its elapsed time from the start of each leg, so its `t` jumps
back to zero between the demo's moves; a time-based window never expired against
that and the plot grew without bound and folded over itself. Checked by feeding
1800 samples across three legs with restarting clocks: the history holds exactly
500 and the axis runs 1301 → 1800.

`impedance_control/telemetry.py` is the sender: one JSON datagram per control step
over UDP to `127.0.0.1:9870`. UDP because a datagram nobody is listening for is
simply dropped — `sendto()` does not block, raise or queue — so the command loop
runs identically whether or not a viewer is attached, and the viewer can be killed
and restarted mid-trajectory. The socket is deliberately left *unconnected*:
`connect()` would make the kernel report ICMP port-unreachable as `ECONNREFUSED`
on the next send, turning "no viewer running" into an exception inside the control
loop. The viewer imports no hardware module, so it cannot command the arm.

Checked end to end with `--replay`: 100 Hz in, arm moving in the browser, the
0.05 s lag built into the replay showing up as the expected ~0.015 rad error.
**Not yet run against the arm.**

## Current gains

Stiff base, compliant wrist, chosen for ergodic exploration:

| joints | kp | kd | why |
|---|---|---|---|
| 1–3 | 10.0 | 0.8 | vendor factory defaults |
| 4–6 | 1.0 | ~0.1 | compliant wrist |

Gives roughly 54–451 N/m at the tip translationally and 0.76–11.1 Nm/rad
rotationally: the base holds position while the wrist gives angularly, which is
what lets a misaligned peg self-align. The wrist gain is almost purely an angular
knob — translational stiffness barely moves with it.

Note the factory `kd = 0.8` is well below critical on joints 1–3 (ζ ≈ 0.22 on j2),
so expect overshoot if anything ever commands a step. There is headroom to 5.0.

---

## Known gaps

- **Host FK has never been checked against the real arm.** A systematic error such
  as the wrong URDF variant (~14 mm) would still be undetected.
- **The URDF inertias are unvalidated** — SolidWorks export values, nothing weighed.
  Every feedforward figure inherits that.
- **No friction model.** `rnea` is rigid-body only. For a contact task this is the
  most significant gap, since joint friction is indistinguishable from tip contact
  force unless modelled.
- **The joint driver firmware is a black box.** The control law in `CLAUDE.md` is
  documented, not verified; the arm held a pose better than that law predicts,
  which suggests it does more than we know.
- `mit2can_park()` now travels with `move_to()` instead of a single one-shot MIT
  setpoint, so the arm should already be at the park pose when MOVE J takes over
  and the handover swing should be small. **Not yet run on hardware** — the
  0x151 handover itself is unchanged, but the approach to it is new.

## Next

Self-collision filtering (the MoveIt SRDF with its 26 `disable_collisions` pairs is
already vendored), then the ergodic controller.
