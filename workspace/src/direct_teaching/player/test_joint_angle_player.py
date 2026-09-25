"""Offline checks of JointAnglePlayer: python test_joint_angle_player.py."""

import tempfile
from pathlib import Path

import numpy as np

from direct_teaching.player.joint_angle_player import JointAnglePlayer

T_RECORDED = np.array([2.0, 2.1, 2.3])  # need not start at 0
Q_RECORDED = np.array([[0.0, 1.0, -1.0, 0.0, 0.5, 0.0], [0.1] * 6, [0.3] * 6])


def _load(q_start: np.ndarray, approach_speed: float = 0.3) -> JointAnglePlayer:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "recording.npz"
        np.savez(path, t=T_RECORDED, q=Q_RECORDED)
        return JointAnglePlayer.load(path, q_start, approach_speed)


def test_approach_lasts_slowest_joint_at_approach_speed() -> None:
    q_start = Q_RECORDED[0] + np.array([0.0, 0.6, 0.0, -0.9, 0.0, 0.0])
    player = _load(q_start)
    t_approach = 0.9 / 0.3
    np.testing.assert_allclose(player.t, [0.0, t_approach, t_approach + 0.1, t_approach + 0.3])
    np.testing.assert_array_equal(player.q[0], q_start)
    np.testing.assert_array_equal(player.q[1:], Q_RECORDED)
    assert player.duration == player.t[-1]


def test_short_approach_lasts_at_least_one_second() -> None:
    player = _load(Q_RECORDED[0] + 0.01)
    assert player.t[1] == 1.0


def test_interpolates_linearly_per_joint() -> None:
    q_start = Q_RECORDED[0] + 0.9
    player = _load(q_start)
    np.testing.assert_allclose(player.joint_angles_at(1.5), (q_start + Q_RECORDED[0]) / 2)
    t_second = player.t[2]
    np.testing.assert_allclose(
        player.joint_angles_at(t_second + 0.1), (Q_RECORDED[1] + Q_RECORDED[2]) / 2
    )


def test_holds_first_and_last_sample_outside_timeline() -> None:
    q_start = Q_RECORDED[0] + 0.5
    player = _load(q_start)
    np.testing.assert_array_equal(player.joint_angles_at(-1.0), q_start)
    np.testing.assert_array_equal(player.joint_angles_at(player.duration + 10.0), Q_RECORDED[-1])


def test_derivatives_match_a_quadratic_sampled_finely() -> None:
    # A central difference is exact on a quadratic, so q = 0.5*a*t^2 checks both.
    a = np.array([0.4, -0.3, 0.2, -0.1, 0.5, 0.0])
    t = np.arange(0.0, 4.0, 0.001)
    player = JointAnglePlayer(t=t, q=0.5 * a * t[:, None] ** 2)
    np.testing.assert_allclose(player.joint_velocities_at(2.0), a * 2.0, rtol=1e-4)
    np.testing.assert_allclose(player.joint_accelerations_at(2.0), a, rtol=1e-3)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
