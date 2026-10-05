# Phase labels on the PiPER: implementation summary

Branch `explore/master-backwards-search`. The work is uncommitted.

## Goal

This is the first step towards testing phase-augmented ergodic control (2-D concept in
`ergodic_controller/Ergodic_Exploration_phase_key_in_lock_2D.ipynb`) on the arm. It does three things:

- records a master trajectory of the task;
- labels every taught datapoint with a phase φ, by projecting it onto that master;
- shows, with a slider over φ, which end-effector positions the phase-conditioned target density holds.

No control loop is changed. Nothing in this step closes a loop on the arm.

## Design decisions

- **Master recorder:** `record_joint_angles.py`, unchanged. A master is one continuous `joint_angles_*.npz`
  demonstration of the whole task, recorded at 100 Hz under gravity compensation and dwell-trimmed by
  `load_recording`.
- **Phase:** the master's normalised time, φ = t / T. A pause while pressing or turning keeps its own share of φ,
  whereas a phase by arc length would collapse it.
- **State:** the E2T2 state `[p, Log_mu(quat)]` of `peg_tcp`. `mu` is the datapoints' mean orientation and serves as one
  tangent space for both the master and the datapoints. This is the same state map as `PoseDistribution`, without a GMM fit.
- **Projection metric:** each axis is divided by its standard deviation over the datapoints, the 6-D analogue of the
  toy's θ/10. A datapoint gets the φ of its nearest master sample, the first one on ties.
- **Derived parameters**, as in the toy:
  - h = the median distance to the k-th nearest neighbour, k = round(√M);
  - dℓ/dφ = the master's scaled path length;
  - σ_f = lead = h / (dℓ/dφ).
- **Phase kernel:** asymmetric around min(1, φ + lead), width σ_f ahead and σ_b behind, normalised per φ.

## Files

| file | what |
|---|---|
| `direct_teaching/distribution/phase_projection.py` (new) | `tcp_poses`, `pose_states`, `master_phases`, `project`, `neighbour_distance`, `phase_weights`, `phase_context`, `save_phase_labels`; hardware-free |
| `direct_teaching/distribution/test_phase_projection.py` (new) | 6 offline checks |
| `label_datapoint_phases.py` (new) | writes `<datapoints>_phase.npz` (`q`, `phi`) beside the input; the input is only read; hardware-free |
| `visualization/visualizer.py` | `Visualizer.show_phase_distribution(master, labelled)` |
| `visualization/visualize_phase_distribution.py` (new) | thin launcher |
| `simulation/meshcat_scene.py` | `animate_phase_distribution`: one cloud per φ, switched by the animation scrubber |

The labelled file has no `t`, so `load_recording` and the existing pipeline still read it as an ordinary datapoint set.

**The view (MeshCat).** All datapoints are drawn in black and the master's TCP path in red. There are 101 φ frames. At
each one the arm stands at the master's pose for that φ, and only that frame's datapoints are shown, coloured by their
target weight with no stall (σ_b = σ_f). The slider is **Animations > default > time**, and it reads φ directly.

## Verification (offline)

- **`test_phase_projection.py`:** 6/6 pass.
  - Master samples project to their own φ.
  - Noisy points land within 2 samples of their true φ.
  - The weights sum to 1 and peak one lead ahead.
  - h matches a hand count.
  - Each state maps back to the FK rotation within 1e-9.
  - A file round trip, from a joint-space master through labelling to `load_recording`.
- **Full offline test suite:** 50 passed, 0 failed. **Lint:** clean on all changed files.
- **Smoke run:** a pseudo-master interpolated through the 8 poses of `datapoints_20261002_171103.npz` in taught order
  labelled them exactly i/7 (0, 0.143, …, 1). The viewer built in 1.2 s. Both smoke files were deleted afterwards.

## Status and open points

- **No real master recorded yet.** Joint 6 was outside its limits, reading about 1092° (three extra turns of the
  wrist), and therefore the arm could not be connected. It has to be brought back within ±120° before recording.
- **The φ slider is unchecked in a browser.** It relies on three.js boolean `visible` tracks to switch the clouds.
- **The std metric has no floor.** An axis that barely varies across the datapoints would dominate the projection.
  Check the per-axis std that `label_datapoint_phases.py` prints on the first real run.
- **The smoke run's σ_f = 0.26 is meaningless:** 8 datapoints make h large.

## How to run (inside the container, from `workspace/src`)

```bash
python record_joint_angles.py   # master: one clean insert-and-turn (touches the arm)
python label_datapoint_phases.py ../output/joint_angles_<ts>.npz ../output/datapoints_<ts>.npz
python visualization/visualize_phase_distribution.py \
    ../output/joint_angles_<ts>.npz ../output/datapoints_<ts>_phase.npz
```
