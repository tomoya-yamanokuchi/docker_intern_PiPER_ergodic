---
paths:
  - "workspace/src/main.py"
  - "workspace/src/impedance_control/**"
  - "workspace/src/test_*.py"
  - "workspace/src/print_joint_limits.py"
---

# Hardware boundary

You are editing code that can command a physical arm.

- Never execute this code. Write it, explain how to run it, and hand it back to me.
- Motion commands are rate-limited and joint-limited in code, not by convention. If a limit is not enforced in the function you are writing, say so in your response.
- Before any motion, the code prints the target and the current state. No silent moves.
- `piper_sdk` calls stay in a thin adapter layer (`impedance_control/`). Control logic lives in `kinematics/` and is testable with the adapter replaced by a recorded or simulated one. If you find yourself putting an SDK call inside a control loop outside `impedance_control/`, that is the bug.
- On any error path, the safe action is to stop the arm, not to retry.
