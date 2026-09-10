#!/usr/bin/env python3
# -*-coding:utf8-*-


def set_mit_mode(interface, quiet=False):
    """
    Switch to MIT control mode. quiet silences the log for streamed commands.
    """
    interface.ModeCtrl(
        0x01,  # CAN command control mode
        0x04,  # MOVE M / MIT
        0,  # speed percentage is not used here
        0xAD,  # MIT mode
    )

    if not quiet:
        print("INFO: Succesfully switched to MIT control mode")


def compose_mit_targets(joint_angles, tuning, qdot_ref=None, tau_ff=None):
    """
    Attach the per-joint tuning constants to a {joint: q} dict, giving the
    {joint: {q, qdot_ref, kp, kd, tau_ff}} form send_mit() expects.

    qdot_ref and tau_ff, when given as {joint: value} dicts, replace the
    constants in tuning for that joint. A streamed move needs both: the velocity
    it is actually meant to be travelling at, and the torque its own dynamics
    require at that point of the path.
    """
    overrides = {"qdot_ref": qdot_ref, "tau_ff": tau_ff}

    return {
        joint: {
            "q": q,
            **tuning[joint],
            **{name: values[joint] for name, values in overrides.items() if values is not None},
        }
        for joint, q in joint_angles.items()
    }


def check_targets(interface, targets):
    """
    Raise if any MIT field is outside its allowed range.

    JointMitCtrl bypasses the check move_j gets: init_soft_joint_limit_on() only
    clamps JointCtrl and the feedback parsing, never MIT. So the limits have to
    be applied by hand here.

    The q limits come from the SDK itself rather than a copied table, so
    they stay correct if SetSDKJointLimitParam() is ever used. The gain limits
    are the ranges JointMitCtrl documents; FloatToUint() does not clamp, so an
    out-of-range gain wraps through the bit mask and reaches the joint as a
    plausible-looking but wrong value instead of raising.
    """
    bad = []

    for joint, target in targets.items():
        j_min, j_max = interface.GetSDKJointLimitParam(f"j{joint}")

        for name, value, v_min, v_max in (
            ("q", target["q"], j_min, j_max),
            ("qdot_ref", target["qdot_ref"], -45.0, 45.0),
            ("kp", target["kp"], 0.0, 500.0),
            ("kd", target["kd"], -5.0, 5.0),
            ("tau_ff", target["tau_ff"], -18.0, 18.0),
        ):
            if not v_min <= value <= v_max:
                bad.append(f"joint {joint} {name}: {value:.4f} outside [{v_min:.4f}, {v_max:.4f}]")

    if bad:
        raise ValueError("MIT target out of range -- " + "; ".join(bad))


def send_mit(interface, targets, quiet=False):
    """
    Send one MIT command per joint. quiet silences the per-call logging, which
    is unreadable when a trajectory streams these at 100 Hz.
    """
    check_targets(interface, targets)

    set_mit_mode(interface, quiet)

    for joint, target in targets.items():
        interface.JointMitCtrl(
            joint,
            target["q"],
            target["qdot_ref"],
            target["kp"],
            target["kd"],
            target["tau_ff"],
        )

    if not quiet:
        print("INFO: Succesfully sent targets: ", targets)
