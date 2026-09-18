#!/usr/bin/env python3
"""Ergodic exploration of a recording's pose distribution on the live arm.

Fits the time-weighted PoseDistribution to a recording, computes the E2T2
coefficients, approaches the recording's first pose in joint space, and then
lets the ergodic controller move a Cartesian target that the main_tast_imp.py
impedance law tracks at 100 Hz. Ctrl-C hands the arm to a position hold.

Run from workspace/src:  python run_ergodic_pipeline.py recording.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from controller.task_imp_controller import CartesianImpedanceController
from direct_teaching.distribution.pose_distribution import PoseDistribution
from direct_teaching.player.joint_angle_player import JointAnglePlayer
from direct_teaching.recorder.joint_angle_recorder import load_recording
from ergodic_controller.ergodic_controller import ErgodicController
from execution.executor_helpers import (
    apply_joint_torques,
    connect_arm,
    hold_current_pose,
    read_joint_velocities,
)

AGX_REFERENCE = Path(__file__).resolve().parent / "agx_reference"
URDF_PATH = AGX_REFERENCE / "piper/piper/urdf/piper_description.urdf"

CONTROL_FREQ_HZ = 100.0
TCP_FRAME_NAME = "peg_tcp"
APPROACH_SPEED = 0.3  # rad/s, as JointAnglePlayer

# As visualization/visualizer.py: 8 components as in the E2T2 paper, Fourier
# modes and quadrature points per dimension from the E2T2 notebook.
N_COMPONENTS = 8
ERGODIC_K = 5
ERGODIC_N = 10
U_MAX = 0.5  # cube units per second

R_WORLD_BASE = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()


def make_controller(dofs: int) -> CartesianImpedanceController:
    """CartesianImpedanceController with the gains validated in main_tast_imp.py."""
    controller = CartesianImpedanceController(
        urdf_path=str(URDF_PATH), dofs=dofs, frame_name=TCP_FRAME_NAME
    )
    controller.set_joint_torque_weights(np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]))
    controller.set_cart_params(
        b=np.array([5.0, 5.0, 5.0, 0.2, 0.2, 0.2]),
        k=np.array([200.0, 200.0, 200.0, 3.0, 3.0, 3.0]),
    )
    return controller


def fit_distribution(
    controller: CartesianImpedanceController, t: np.ndarray, q: np.ndarray
) -> PoseDistribution:
    poses = [controller.pin_model.forward_kinematics(q_i, TCP_FRAME_NAME) for q_i in q]
    p = np.array([p_i for p_i, _ in poses])
    rotations = np.array([R_i for _, R_i in poses])
    return PoseDistribution(t, p, rotations, N_COMPONENTS)


def prepare_exploration(
    recording: Path,
) -> tuple[CartesianImpedanceController, PoseDistribution, ErgodicController, np.ndarray]:
    """Everything offline, before the arm is touched; returns the recording's first q (6,) rad last."""
    controller = make_controller(dofs=6)
    t_rec, q_rec = load_recording(recording)
    distribution = fit_distribution(controller, t_rec, q_rec)
    print(f"computing ergodic coefficients for {recording}")
    ergodic = ErgodicController(distribution.pdf, 6, ERGODIC_K, ERGODIC_N, U_MAX)
    return controller, distribution, ergodic, q_rec[0]


def make_approach(q_start: np.ndarray, q_end: np.ndarray) -> JointAnglePlayer:  # (6,) rad
    """Joint-space line from the current pose, timed as JointAnglePlayer.load's approach.

    Impedance control is entered at the current pose; a distant Cartesian target
    would jerk the arm to the recording's start.
    """
    t_approach = max(1.0, np.max(np.abs(q_end - q_start)) / APPROACH_SPEED)
    return JointAnglePlayer(t=np.array([0.0, t_approach]), q=np.vstack([q_start, q_end]))


def main(recording: Path) -> None:
    controller, distribution, ergodic, q_first = prepare_exploration(recording)
    x_ref = distribution.pose_to_state(
        *controller.pin_model.forward_kinematics(q_first, TCP_FRAME_NAME)
    )

    robot = connect_arm()
    joint_angles = np.array(robot.get_joint_angles().msg)
    approach = make_approach(joint_angles, q_first)

    period = 1.0 / CONTROL_FREQ_HZ
    print(
        f"q = {np.round(joint_angles, 4)} rad\n"
        f"approaching start pose for {approach.duration:.1f} s, then exploring; Ctrl-C to stop"
    )
    try:
        t0 = time.monotonic()
        while True:
            start_time = time.monotonic()
            t = start_time - t0
            if t < approach.duration:
                p_target, R_target = controller.pin_model.forward_kinematics(
                    approach.joint_angles_at(t), TCP_FRAME_NAME
                )
            else:
                x_ref = ergodic.step(x_ref, period)
                p_target, R_target = distribution.state_to_pose(x_ref)
            joint_angles = run_control_cycle(robot, controller, p_target, R_target, R_WORLD_BASE)

            elapsed_time = time.monotonic() - start_time
            if elapsed_time < period:
                time.sleep(period - elapsed_time)
            else:
                print(f"warning: control loop overrun {elapsed_time:.3f}s > {period:.3f}s")

    except KeyboardInterrupt:
        print("\ninterrupted, holding current pose")
    finally:
        hold_current_pose(robot, joint_angles)
    if ergodic.step_count:
        print(f"ergodic metric: {ergodic.ergodic_metric():.4f}")


def run_control_cycle(
    robot,
    controller: CartesianImpedanceController,
    p_target: np.ndarray,  # (3,) m
    R_target: np.ndarray,  # (3, 3)
    R_world_base: np.ndarray,  # (3, 3)
) -> np.ndarray:  # (6,) rad, the q the torque was computed from
    """Read the state, send the torque pulling the TCP to (p_target, R_target)."""
    joint_angles = np.array(robot.get_joint_angles().msg)
    joint_velocities = read_joint_velocities(robot)
    cmd_torque = controller.compute_cartesian_torque(
        desired_pos=p_target,
        desired_ori=R_target,
        q_cur=joint_angles,
        v_cur=joint_velocities,
        base_orientation=R_world_base,
    )
    apply_joint_torques(robot, cmd_torque)
    return joint_angles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Mandatory: the arm explores whatever recording this is, so never pick one implicitly.
    parser.add_argument("recording", type=Path, help="joint_angles_*.npz")
    args = parser.parse_args()
    main(args.recording)
