---
paths:
  - "workspace/src/agx_reference/piper/**"
  - "workspace/src/agx_reference/controller/**"
---

# Hardware boundary

You are editing code that can command a physical arm.

- Never execute this code. Write it, explain how to run it, and hand it back to me.
- `agx_reference/` is a byte-identical copy of upstream and is the validated
  working state. Do not reformat it or tidy its style. If a change is genuinely
  needed there, say what it is and why before touching it.
- Motion commands are rate-limited and joint-limited in code, not by convention.
  If a limit is not enforced in the function you are writing, say so in your
  response. This matters more under torque control than it did under position
  control: the firmware's soft limits act on position setpoints and do nothing
  to stop a commanded torque from driving a joint into its stop.
- Before any motion, the code prints the target and the current state. No silent
  moves.
- SDK calls stay at the edge. A control loop reads state and writes torques
  through the `pyAgxArm` handle; the law computing those torques is a pure
  function of state, testable with no arm attached. `controller/` and `core/`
  are that pure half, `piper/main_*.py` are the loops. An SDK call inside the
  torque computation is the bug.
- On any error path, the safe action is to stop the arm, not to retry. "Stop"
  means a position hold at the current angles
  (`move_mit(j, q[j-1], 0, 10, 0.8, 0)`), never zero torque — this arm has no
  brakes and falls the moment torque is removed.
