# PiPER joint limits and MIT command ranges

Reference for anyone editing a target pose or a gain table. Two independent
sources agree on the joint limits, and both are checked below against the
installed SDK rather than copied from a datasheet.

## Joint limits

| joint | rad | deg | note |
|---|---|---|---|
| 1 | −2.6179 … 2.6179 | ±150 | base yaw |
| 2 | 0.0 … 3.14 | 0 … 180 | **one-sided** — shoulder |
| 3 | −2.967 … 0.0 | −170 … 0 | **one-sided** — elbow |
| 4 | −1.745 … 1.745 | ±100 | forearm roll |
| 5 | −1.22 … 1.22 | ±70 | wrist pitch |
| 6 | −2.0944 … 2.0944 | ±120 | wrist roll |

⚠️ **Joints 2 and 3 are one-sided, so the all-zeros pose sits exactly on their
limit.** That is still a valid pose — zero is the parking target this project
uses — but it means joint 2 can only travel positive and joint 3 only negative,
and a target that overshoots by any amount is rejected rather than clipped.

### Where these come from

- `piper_sdk/piper_param/piper_param_manager.py:29-34`, the SDK's own table.
  Read at runtime through `interface.GetSDKJointLimitParam(f"j{joint}")`.
- The URDF, via `PiperKinematics().q_min` / `.q_max`.

The two match to rounding (the SDK writes j6 as 2.09439, the URDF as 2.0944),
so either can be trusted. Prefer the model's `q_min`/`q_max` in code — `kin.clamp(q)`
and `kin.in_limits(q)` already use them — rather than retyping the numbers.

## MIT command ranges

`JointMitCtrl(motor_num, pos_ref, vel_ref, kp, kd, t_ref)`, checked by
`impedance_control.mit.check_targets()` before anything is sent.

| field | range | vendor reference |
|---|---|---|
| `motor_num` | 1 … 6 | — |
| `q` (`pos_ref`) | the joint limits above | — |
| `qdot_ref` (`vel_ref`) | −45.0 … 45.0 rad/s | 0.0 |
| `kp` | 0.0 … 500.0 | 10 |
| `kd` | −5.0 … 5.0 | 0.8 |
| `tau_ff` (`t_ref`) | −18.0 … 18.0 Nm | 0.0 |

⚠️ **These are checked, not clamped, and that check is ours.** `FloatToUint()`
in `piper_sdk/protocol/piper_protocol_base.py:382` scales and truncates without
clamping, and the result is bit-masked into the frame — so an out-of-range gain
would reach the joint as a plausible-looking but wrong value instead of raising.
`check_targets()` exists to turn that into a `ValueError` on this side of the
wire.

Two quantisation limits worth knowing while tuning
(`piper_interface_v2.py:3062-3066`): `tau_ff` is 8 bits over ±18 Nm, i.e. 0.14 Nm
steps, and `kd` caps at 5.0 while `kp` runs to 500 — damping runs out well
before stiffness does.
