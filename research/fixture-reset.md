# Resetting an articulated fixture between trials

Research note for issue #14 (child of #11). Question: how do autonomous
manipulation setups reset a door, drawer or lock between trials without a human,
what goes wrong, and how is a failed reset detected? Read against our setup: a
torque-controlled PiPER under Cartesian impedance, sensorless (joint angles only),
a spring-loaded door that is locked, a key that couples the arm to the door once
turned, a lock that re-engages when the door closes, and a key that comes out only
with the door closed.

Every claim below is sourced to the paper or code that owns it. Sections marked
**Inference** are this note's reading for our task, not something a source says.

## Short answer

- Real-hardware systems do not learn the deterministic part of a reset. They
  script it, check it against a tolerance on the robot's own joint state, retry a
  few times, then stop and ask a human (ROBEL's `scripted_reset.py`; MEDAL++'s
  fixed-joint-position arm reset).
- Learned resets (Leave No Trace, MEDAL/MEDAL++, SERL, MTRF) all need a
  success/state signal for the *fixture* — simulator state, motion capture, or a
  vision classifier. None of them works from joint angles alone, and all of them
  still report needing intermittent human interventions.
- Reversing a demonstration is valid only where the dynamics are reversible
  (TRASS names knob-turning and drawer-closing as such cases); reverse curricula
  assume the simulator can be reset to an arbitrary state.
- No primary source found treats a spring-return fixture that stays coupled to the
  arm through a key and an interlock. The closest is compliant equilibrium-point
  control of doors, which fits the hinge from the end-effector trajectory alone.
- Failed-reset detection in the literature is a necessary-not-sufficient check:
  "the reset policy did not reach the reset set within N attempts" (LNT), "a joint
  is more than 5 degrees from its reset angle after 3 retries" (ROBEL).

## 1. Reset-free / forward-backward learning

**Leave No Trace** (Eysenbach et al., ICLR 2018, [arXiv:1711.06782](https://arxiv.org/abs/1711.06782))
learns a forward and a reset policy together. The reset policy's Q-value gates
the forward policy: if it falls below `Q_min` the action is not taken and the
reset policy takes over ("early abort"). A *hard* (manual) reset is called when
"the reset policy fails to reset after N attempts", which the authors call "a
necessary (but not sufficient) condition" for an irreversible state (§4.2). Larger
N means fewer hard resets but more time stuck. Reported limitations (§7): all
experiments are in simulation; on a real robot "a challenge ... is automatically
identifying when the agent has reset ... we must infer states from noisy sensor
observations"; all manual resets are treated as equally costly. They also show a
"forward-only" agent fails even on a reversible task (ball-in-cup): once it
succeeds it stays at the goal and stops practising (§6.1).

**EARL benchmark** (Sharma et al., ICLR 2022, [arXiv:2112.09605](https://arxiv.org/abs/2112.09605))
formalises autonomous RL with an intervention every `1/epsilon` steps. Its
motivating example is a robot practising closing a door, which must learn to
open it again (§4). Findings: standard RL degrades sharply as resets become rare
(Fig. 2); reset-free methods "fail to make progress on more challenging tasks
compared to methods provided with oracle resets"; without resets the visited
state distribution drifts away from both initial and goal states (Fig. 3). It
defines reversibility as ergodicity of the MDP and gives as its example that "a
static robot arm can be setup such that there always exists an action sequence
to open or close the door". Sawyer-Door is one of its tasks.

**MEDAL** (Sharma et al., ICML 2022, [arXiv:2205.05212](https://arxiv.org/abs/2205.05212))
trains the backward policy to match the state distribution of the forward
demonstrations, via a discriminator; backward demonstrations are optional
(Algorithm 1). Simulation only.

**MEDAL++** (Sharma et al., ICRA 2023, [arXiv:2303.01488](https://arxiv.org/abs/2303.01488))
takes it to a real Franka with wrist-camera image classifiers for both forward
success and backward (reset) states. Relevant details:
- Forward and backward demonstrations were teleoperated "together, one after the
  other" (App., Teleoperation) — i.e. a separately taught reset.
- "we also reset the arm every 1000 steps to avoid hitting singular positions ...
  the controller just resets the arm to a fixed joint position" (App.) — a
  scripted arm reset alongside the learned fixture reset.
- Limitation: "intermittent human interventions to reset the environment can be
  important to learn successfully. The robotic system can get stuck in a specific
  state ... due to poor exploration, even if that state itself is not
  irreversible" (§VI).

**R3L / Ingredients of real-world RL** (Zhu et al., ICLR 2020, [arXiv:2004.12570](https://arxiv.org/abs/2004.12570)):
reset-free training from low-dimensional state gets high *training* reward but
poor *test* performance because "the agent spends all its time stuck at the
goal"; "the reset encodes some prior information about the task" (§3). Learned reset
controllers that return to one initial state give policies that "depend heavily
on a narrow range of initial states" (§4.1). Their alternative is a novelty-seeking
perturbation controller instead of a reset.

**MTRF** (Gupta et al., ICRA 2021, [arXiv:2104.11203](https://arxiv.org/abs/2104.11203)):
several tasks reset each other, chosen by a hand-written *task graph* `G(s)` that
maps the current state to the task to run, using threshold predicates such as
`|q - q_waypoint| < 0.05` (App. D, Alg. 3). 60 h on hardware without human
intervention — but the object state came from a motion-capture system.

**SERL** (Luo et al., ICRA 2024, [arXiv:2401.16013](https://arxiv.org/abs/2401.16013))
packages forward-backward training (two agents, each with its own reward
classifier, §4.3). Its impedance controller section is the part most relevant to
a spring: the RL target at 10 Hz is tracked by a 1 kHz impedance law, and the
reference is **clipped at the real-time layer** so the pose error, and hence the
spring force, stays bounded "whenever in contact", without blocking fast
free-space motion (§4.5, Fig. 3). They prefer this to a force/torque threshold,
citing F/T noise and calibration.

**PAINT** (Xie et al., NeurIPS 2022, [arXiv:2210.10765](https://arxiv.org/abs/2210.10765))
makes the agent request help itself: a classifier predicts irreversibility,
trained from labels obtained by binary search over a trajectory (O(log T) queries),
since every state after an irreversible one is irreversible. Simulation only.

## 2. Reversing a demonstration

**TRASS** (Nair et al., ICRA 2020, [arXiv:1810.01128](https://arxiv.org/abs/1810.01128))
explores outward from goal states and learns the time-reversed trajectories. It
lists where reversal applies: "there are some cases where even when the gripper is
in the state, the dynamics are reversible and TRASS can be applied, for example
turning a knob or closing a drawer" (Preliminaries). It cannot apply where the
robot-object interaction is irreversible (holding an object up) or the object
change is irreversible (Preliminaries, §VIII).

**Reverse curriculum generation** (Florensa et al., CoRL 2017, [arXiv:1707.05300](https://arxiv.org/abs/1707.05300))
includes a key-insertion-and-turn task, but its Assumption 1 is "we can
arbitrarily reset the agent into any start state" (§3.3) — a simulator property.

**Inference.** A spring-loaded door is *not* time-reversible under our control:
pushing it open is work against the spring, while closing it is the spring doing
work on the arm. Replaying the opening trajectory backwards gives the right
*path*, but the force balance is inverted, so the replay is only safe if the arm
tracks it compliantly and with a bounded error (SERL's clipping), letting the
spring lead rather than the arm push.

## 3. Separately taught or scripted resets

**ROBEL** (Ahn et al., CoRL 2019, [arXiv:1909.11639](https://arxiv.org/abs/1909.11639);
code [`robel/dclaw/scripted_reset.py`](https://github.com/google-research/robel/blob/master/robel/dclaw/scripted_reset.py))
is the clearest real-hardware reset in the set, run over thousands of hours. The
turned valve is itself on a Dynamixel (`object`, motor 50), so the fixture is
reset by actuating it, not by the hand. The claw reset is a staged script:
disengage some joints to avoid tangling, move, then check each motor against
`GOOD_ERROR_TOL = 5 deg`; on failure log the bad motors, run a disentangle
sequence, retry; after `MAX_RESET_RETRIES = 3` print "Please fix motors" and
block on `input()`. The paper also reports 29 broken motors over ~14,000 training
hours across two sites (Table 3).

**MEDAL++** (above): taught forward and backward demonstrations, plus a scripted
fixed-joint arm reset.

**Gu et al.** (ICRA 2017, [arXiv:1610.00633](https://arxiv.org/abs/1610.00633)):
in simulation "the door has a spring such that it closes gradually when no force
is applied" and a latch that opens only past a handle angle. On the real robots
"we cannot automatically reposition the door for every episode, so the pose of
the door was kept fixed", and the door angle came from an IMU on the door (§VI).
The paper does not say how the real door returned to closed between episodes.

## 4. Compliant tracking of a moving fixture

**Sturm, Jain, Stachniss, Kemp, Burgard** (IROS 2010,
[PDF](https://cvai.cit.tum.de/_media/spezial/bib/sturm10iros.pdf)) open doors and
drawers with a low-impedance arm under equilibrium-point control. The articulation
model (rotational vs prismatic, centre, radius) is fitted online from the
**end-effector trajectory alone**, and the next equilibrium point is stepped 1 cm
along the estimated mechanism Jacobian. They note the commanded equilibrium path
"will be different from the end effector trajectory ... unless there are no
external forces other than gravity" (§III) — that gap is the force signal. The
term keeping the hook on the handle did use a wrist F/T sensor (5 N). 37/40
successes; all three failures were missing the handle; "the hook could slip off
the handle if a wrong model had been estimated" (§IV-A).

No primary source was found for compliant tracking of a fixture's *spring return*
as a reset step.

## 5. How a failed reset is detected

| source | signal | rule | then |
|---|---|---|---|
| LNT | simulator state vs set `S_reset` | not reached in N reset attempts | manual reset |
| LNT | reset policy's Q-value | `Q < Q_min` during forward run | abort to reset policy |
| ROBEL | per-joint encoder angle | any joint > 5 deg from target | disentangle, retry ×3, then wait for Enter |
| MTRF | motion-capture object state | threshold predicates in task graph | pick the recovery task |
| MEDAL++ / SERL | wrist-camera classifier | learned success / reset classifier | continue; human intervenes intermittently |
| PAINT | learned irreversibility classifier | predicted irreversible | ask for help |
| Gu et al. | IMU on the door | door angle | (not reported) |

Only ROBEL's check uses nothing but the robot's own joint encoders — and it
checks the robot, not the fixture; the fixture there has its own encoder.

## 6. Failure modes reported

- **Stuck at the goal** without resets: the agent stops practising the hard part
  (LNT §6.1, R3L §3). For us: an open door with the arm still coupled is that goal.
- **Stuck in a reversible state** from poor exploration, needing a human anyway
  (MEDAL++ §VI).
- **State drift** away from initial and goal distributions without resets (EARL
  Fig. 3).
- **Narrow initial-state support** when a learned reset always returns to one
  state (R3L §4.1).
- **Irreversible states** the reset cannot undo (LNT, PAINT, EARL), detected only
  after N failed attempts.
- **Real-world reset detection from noisy sensors** is unsolved in LNT (§7).
- **Mechanical**: tangling and motors failing to reach the reset angle (ROBEL
  script); motor breakage under long autonomous runs (ROBEL Table 3); missing the
  handle, hook slip on a wrong articulation model (Sturm 2010).
- **Controller**: an over-stiff tracker damages the part in contact, an
  over-compliant one is slow; unbounded reference error in contact (SERL §4.5).

## 7. What this means for the locked door (inference)

Not established by any source; it is this note's mapping of the above.

1. **Script the reset; do not learn it.** Every real setup that reset a fixture
   repeatably scripted the deterministic part. Our sequence is deterministic and
   has a hard ordering constraint (door closed before key turns back, key turned
   back before withdrawal), which fits a ROBEL-style staged script with a check
   after each stage.
2. **Closing stage: let the spring lead.** While the key couples the arm, track the
   reversed path of the trial's own opening (or a taught closing path) under low
   stiffness with the reference clipped to a bounded error, SERL-style, so the arm
   neither fights nor drops the door.
3. **Detect each stage from joint angles against a taught reference.** Teach the
   checkpoint poses once (door closed with key turned, key turned back, key out,
   start pose) and check `|q - q_ref|` per joint with a tolerance, ROBEL-style.
   Because the door is coupled through the key, the arm's own pose *is* the door's
   pose at stages 1-2, which is what makes a sensorless check possible there.
4. **Detect a jammed stage by stall.** If the door is not fully latched the key
   will not come out; under impedance the withdrawal target runs away from the
   measured pose. The commanded-versus-measured gap is the same signal Sturm et al.
   use as an implicit force. Bound it, and treat "gap above threshold and no
   progress for T seconds" as a failed stage.
5. **Retry a bounded number of times, then hold and ask.** ROBEL retries 3 times
   and waits for Enter; LNT uses N attempts. With an operator present but
   hands-off, a held pose plus a prompt is the natural escalation.

## Not verified

- Gu et al. do not state how the real door closed between episodes; the spring
  is stated only for the simulated door.
- ROBEL's object-motor reset was read from the released code, not from the paper
  text, which does not describe the valve reset.
- MTRF's motion-capture use was read from the experimental-setup text; whether
  every hardware predicate used it was not checked.
- No source was found that resets a spring-return fixture by compliant tracking,
  nor one with a key/lock interlock coupling the arm to the fixture. Section 7 is
  therefore inference.
- Full texts were read via arXiv PDFs; figures and supplementary videos were not
  checked.
