"""Offline checks of JointAngleRecorder: python test_joint_angle_recorder.py."""

import tempfile
from pathlib import Path

import numpy as np

from direct_teaching.recorder.joint_angle_recorder import (
    JointAngleRecorder,
    load_recording,
    trim_dwell,
)


def test_samples_on_fixed_grid() -> None:
    recorder = JointAngleRecorder(rec_freq_hz=10.0)
    # A 200 Hz loop starting at an arbitrary clock value, for exactly 1 s.
    for k in range(200):
        recorder.sample(1000.0 + k / 200.0, np.full(6, k))
    assert len(recorder) == 10
    # Each slot takes the first loop tick at or after it. Subtracting a large
    # clock value can round a tick that is exactly on a slot to just before it,
    # so that slot goes to the next tick: late by at most one tick, never early.
    t = np.array(recorder._t)
    slots = np.arange(10) / 10.0
    assert np.all(t >= slots - 1e-9)
    assert np.all(t < slots + 1 / 200 + 1e-9)
    ticks = np.array(recorder._q)[:, 0]
    np.testing.assert_allclose(t, ticks / 200.0, atol=1e-9)


def test_overrun_drops_samples_instead_of_bunching() -> None:
    recorder = JointAngleRecorder(rec_freq_hz=10.0)
    for t in (0.0, 0.1, 0.45, 0.46, 0.49, 0.5):
        recorder.sample(t, np.zeros(6))
    # 0.2, 0.3 and 0.4 are lost; 0.46 and 0.49 are still inside the 0.4 slot.
    np.testing.assert_allclose(recorder._t, [0.0, 0.1, 0.45, 0.5])


def test_save_load_round_trip() -> None:
    recorder = JointAngleRecorder(rec_freq_hz=10.0)
    rng = np.random.default_rng(0)
    q = rng.uniform(-1.0, 1.0, (5, 6))
    for k in range(5):
        recorder.sample(0.1 * k, q[k])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "recording.npz"
        recorder.save(path)
        t_loaded, q_loaded = load_recording(path)
    np.testing.assert_allclose(t_loaded, 0.1 * np.arange(5))
    np.testing.assert_array_equal(q_loaded, q)


def test_sample_copies_q() -> None:
    recorder = JointAngleRecorder(rec_freq_hz=10.0)
    q = np.zeros(6)
    recorder.sample(0.0, q)
    q[0] = 1.0
    assert recorder._q[0][0] == 0.0


def test_empty_recording_saves_with_six_columns() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "empty.npz"
        JointAngleRecorder(rec_freq_hz=10.0).save(path)
        with np.load(path) as data:
            t, q = data["t"], data["q"]
    assert t.shape == (0,)
    assert q.shape == (0, 6)


def _dwell_recording() -> tuple[np.ndarray, np.ndarray]:
    # 4 samples still at q_a, 3 moving, 5 still at q_b, at 10 Hz.
    rng = np.random.default_rng(0)
    q_a, q_b = rng.uniform(-1.0, 1.0, (2, 6))
    moving = rng.uniform(-1.0, 1.0, (3, 6))
    q = np.vstack([np.tile(q_a, (4, 1)), moving, np.tile(q_b, (5, 1))])
    return 0.1 * np.arange(len(q)), q


def test_trim_dwell_keeps_motion_and_its_still_ends() -> None:
    t, q = _dwell_recording()
    q[1, 0] += 0.5e-3  # encoder jitter inside the tolerance
    t_trimmed, q_trimmed = trim_dwell(t, q)
    np.testing.assert_array_equal(q_trimmed, q[3:8])
    np.testing.assert_allclose(t_trimmed, 0.1 * np.arange(5))


def test_trim_dwell_leaves_a_recording_without_dwell() -> None:
    rng = np.random.default_rng(1)
    q = rng.uniform(-1.0, 1.0, (6, 6))
    t_trimmed, q_trimmed = trim_dwell(0.1 * np.arange(6), q)
    np.testing.assert_array_equal(q_trimmed, q)
    np.testing.assert_allclose(t_trimmed, 0.1 * np.arange(6))


def test_trim_dwell_rejects_a_still_recording() -> None:
    q = np.tile(np.ones(6), (10, 1))
    try:
        trim_dwell(0.1 * np.arange(10), q)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_load_recording_trims_dwell() -> None:
    t, q = _dwell_recording()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "recording.npz"
        np.savez(path, t=t, q=q)
        _, q_loaded = load_recording(path)
    np.testing.assert_array_equal(q_loaded, q[3:8])


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
