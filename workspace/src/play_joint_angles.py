#!/usr/bin/env python3
"""Replay a joint-angle recording under Cartesian impedance control.

The loop is agx_reference/piper/main_tast_imp.py with its gains, except that the
Cartesian target moves, set each cycle to the flange pose of the recorded q at
that time, and that the inertial and friction torque of the recorded motion is
fed forward on top. It first approaches the recording's start pose from the
current one, and after the last sample it holds the final pose until Ctrl-C,
which hands the arm to a position hold.

Run from workspace/src:  python play_joint_angles.py recording.npz
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from controller.feed_forward import FeedForward
from controller.task_imp_controller import CartesianImpedanceController
from direct_teaching.player.joint_angle_player import JointAnglePlayer
from direct_teaching.player.tracking_error import (
    compute_tracking_errors,
    save_tracking_error_plot,
)
from execution.executor_helpers import (
    apply_joint_torques,
    connect_arm,
    hold_current_pose,
    read_joint_velocities,
)

AGX_REFERENCE = Path(__file__).resolve().parent / "agx_reference"
URDF_PATH = AGX_REFERENCE / "piper/piper/urdf/piper_description.urdf"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

CONTROL_FREQ_HZ = 200.0
EE_FRAME_NAME = "link6"


def make_controller(dofs: int) -> CartesianImpedanceController:
    """CartesianImpedanceController with the gains validated in main_tast_imp.py."""
    controller = CartesianImpedanceController(
        urdf_path=str(URDF_PATH), dofs=dofs, frame_name=EE_FRAME_NAME
    )
    controller.set_joint_torque_weights(np.array([1.0, 1.0, 1.0, 0.5, 1.0, 0.5]))
    controller.set_cart_params(
        b=np.array([5.0, 5.0, 5.0, 0.2, 0.2, 0.2]),
        k=np.array([200.0, 200.0, 200.0, 5.0, 5.0, 5.0]),
    )
    return controller


def main(recording: Path) -> None:
    robot = connect_arm()
    joint_angles = np.array(robot.get_joint_angles().msg)

    controller = make_controller(robot.joint_nums)
    feed_forward = FeedForward(urdf_path=str(URDF_PATH), dofs=robot.joint_nums)
    player = JointAnglePlayer.load(recording, q_start=joint_angles)

    R_world_base = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()
    period = 1.0 / CONTROL_FREQ_HZ

    print(f"q = {np.round(joint_angles, 4)} rad")
    print(f"playing {recording} for {player.duration:.1f} s incl. approach, then holding")
    print("Ctrl-C to stop")
    log = []  # (t s, q (6,) rad) per cycle; the target is recomputed from t
    try:
        t0 = time.monotonic()
        while True:
            start_time = time.monotonic()
            t = start_time - t0

            joint_angles = run_control_cycle(
                robot, controller, feed_forward, player, t, R_world_base
            )
            log.append((t, joint_angles))

            elapsed_time = time.monotonic() - start_time
            if elapsed_time < period:
                time.sleep(period - elapsed_time)
            else:
                print(f"warning: control loop overrun {elapsed_time:.3f}s > {period:.3f}s")

    except KeyboardInterrupt:
        print("\ninterrupted, holding current pose")
    finally:
        # Hold before plotting, so a failed plot cannot leave the arm unheld.
        hold_current_pose(robot, joint_angles)
        if log:
            save_tracking_error(controller, player, log)


def run_control_cycle(
    robot,
    controller: CartesianImpedanceController,
    feed_forward: FeedForward,
    player: JointAnglePlayer,
    t: float,  # s since the loop started
    R_world_base: np.ndarray,  # (3, 3)
) -> np.ndarray:  # (6,) rad, the q the torque was computed from
    """Read the state, send the torque pulling the flange to FK(player at t)."""
    joint_angles = np.array(robot.get_joint_angles().msg)
    joint_velocities = read_joint_velocities(robot)
    x_target, r_target = controller.pin_model.forward_kinematics(
        player.joint_angles_at(t), EE_FRAME_NAME
    )
    cmd_torque = controller.compute_cartesian_torque(
        desired_pos=x_target,
        desired_ori=r_target,
        q_cur=joint_angles,
        v_cur=joint_velocities,
        base_orientation=R_world_base,
    )
    cmd_torque = cmd_torque + feed_forward.compute_torque(
        q_cur=joint_angles,
        qd_cur=joint_velocities,
        qd_des=player.joint_velocities_at(t),
        qdd_des=player.joint_accelerations_at(t),
    )
    apply_joint_torques(robot, cmd_torque)
    return joint_angles


def save_tracking_error(
    controller: CartesianImpedanceController,
    player: JointAnglePlayer,
    log: list[tuple[float, np.ndarray]],
) -> None:
    t, q = (np.array(column) for column in zip(*log, strict=True))
    q_target = np.array([player.joint_angles_at(t_i) for t_i in t])
    position_error, orientation_error = compute_tracking_errors(
        controller.pin_model, q, q_target, EE_FRAME_NAME
    )
    path = OUTPUT_DIR / f"tracking_error_{datetime.now():%Y%m%d_%H%M%S}.png"
    t_recording = (float(player.t[1]), player.duration)
    save_tracking_error_plot(t, position_error, orientation_error, t_recording, path)
    print(f"saved tracking error plot to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Mandatory: the arm replays whatever recording this is, so never pick one implicitly.
    parser.add_argument("recording", type=Path, help="joint_angles_*.npz")
    args = parser.parse_args()
    main(args.recording)
