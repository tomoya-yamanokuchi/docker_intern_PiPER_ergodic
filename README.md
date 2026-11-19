# PiPER Ergodic: sequential ergodic exploration on the AgileX PiPER

Ergodic exploration (E2T2, tensor-train) for sequential contact tasks, such as a
locked door or a covered peg-in-hole, extended with a sliding target and a
forgetting statistic. Runs on a real AgileX PiPER 6-DoF arm, without ROS, in Python 3.10
over SocketCAN.

All live code is in `final/src/`. The vocabulary (master, datapoint, phase, trial and
so on) is defined in [`final/GLOSSARY.md`](final/GLOSSARY.md). Every script's docstring
gives its exact usage.

## Safety first

- **The arm has no holding brakes.** It falls as soon as it loses torque: AC cut,
  E-stop, disable, or a MIT command with `kp = kd = t_ff = 0`. Support it before
  powering down.
- Every arm script runs until **Ctrl-C**, which hands each joint to a position hold so
  the arm stays up. **Enter** during a trial aborts that trial.
- If the arm ignores commands, check it is not in **teaching mode** (the physical button).
- **Firmware flag:** an arm reporting S-V1.8-6 needs `--fw V183` on every arm script;
  S-V1.8-2 needs no flag. The wrong profile mis-scales every torque. With two USB-CAN
  adapters, pick the arm with `--can can1`.

## Setup

| file | purpose |
|---|---|
| `Dockerfile`, `build.sh` | the image (Ubuntu 22.04, `pyAgxArm`, Pinocchio, `ttpy`, meshcat) |
| `run.sh` | starts the container `piper-ergodic` (privileged, host network) |
| `can_activate.sh` | brings up the CAN interface |
| `final/src/run.sh` | runs any script with the right `PYTHONPATH`; always go through it |
| `final/check.sh` | ruff plus all `test_*.py` (hardware-free) |

```bash
# host, once:            sh build.sh
# host, every boot or USB re-plug (the adapter comes up DOWN, the arm needs 1 Mbit/s):
sudo bash can_activate.sh can0 1000000
ip -details link show can0                       # UP, bitrate 1000000
cat /sys/class/net/can0/statistics/rx_packets    # must increase while the arm is powered

sh run.sh                                        # host: start the container
cd /home/jens/workspace/docker_intern_PiPER_ergodic/final/src   # inside it
```

`rx_packets` stuck at 0 means nothing reaches the adapter. In practice that has been a
hung USB-CAN adapter (constant purple LED): re-plug or swap it, then check E-stop, power and
the green CAN screw terminal.

Viewers open in the host browser: MeshCat at `http://localhost:7000`.

## Workflow: one task, start to finish

Every script writes to `final/output/` (gitignored). Copy recordings worth keeping
into `final/saved_recordings/<task>/`. Steps marked **ARM** move the robot.

**1. Record the expert (master). ARM.** The arm is under gravity compensation, so you
backdrive it once through the whole task by hand. Ctrl-C stops and saves.

```bash
sh run.sh record_master.py                       # writes master_<ts>.npz
sh run.sh view_master.py ../output/master_<ts>.npz
```

**2. Teach the datapoints (exploration distribution). ARM.** Pose the arm by hand.
**Enter** confirms the pose as a datapoint and **u** withdraws the last one. To weight
a region, confirm there again after re-posing slightly. MeshCat shows the master and
the fitted density. A second argument extends an earlier file.

```bash
sh run.sh teach_exploration.py ../output/master_<ts>.npz [exploration_<ts>.npz]
```

**3. Label the datapoints (no arm).** Each datapoint gets the phase of its nearest
master sample, in position and orientation, and the output is the task file the
controller runs on. A fixture that does not reset itself, such as the cover of the
covered peg-in-hole, also needs a *closing trajectory*. Record it with
`record_master.py`, starting from the master's start.

```bash
sh run.sh label.py ../output/master_<ts>.npz ../output/exploration_<ts>.npz [closing_<ts>.npz]
sh run.sh view_labels.py ../output/task_<ts>.npz      # check the phase spread it prints
```

**4. Run trials. ARM.** Start with the arm at the master's start, then press Enter, or
use `--auto` for a 2 s settle. After each trial the arm resets itself by replaying the
master, plus the closing trajectory if the task has one.

```bash
sh run.sh run_trials.py <e2t2|expanding|sliding|retry> <task>.npz [n_trials] [--auto] [--out=DIR] [--success-phase=P]
sh run.sh view_run.py ../output/evaluation/run_<method>_<ts>.npz [trial]
sh run.sh analyse_runs.py ../output/evaluation/run_*.npz    # success rates, times
```

`retry` is the proposed method (sliding target, forgetting, retry on a stall), and
`e2t2` is the baseline. The covered peg-in-hole counts as inserted at
`--success-phase=0.8`.

**Without the arm:** `sh run.sh simulate.py [method] [n_trials]` runs the whole chain
on a fake arm.

## Kept tasks

| task (folder in `final/saved_recordings/`) | files (`.npz`) |
|---|---|
| Locked door (`door/`) | master `master_20261021_202500`; experiment task `task_20261023_131342` (exploration `exploration_20261023_131332`); first-success task `task_20261022_205400` (exploration `exploration_20261022_205334`) |
| Covered peg-in-hole (`covered-peg-in-hole/`) | task `task_20261111_194118`, from master `master_20261111_174332`, exploration `exploration_20261111_193938` and closing trajectory `master_20261111_184439` |

Example: `sh run.sh run_trials.py retry ../saved_recordings/door/task_20261023_131342.npz 4 --auto --fw V183`

## Code map (`final/src/`)

| path | contents | touches arm |
|---|---|---|
| `method/law.py` | the control law; `METHODS` lists the methods | no |
| `method/task.py`, `method/pose.py` | labelling, task file, pose metric | no |
| `arm/piper.py` | the only code that calls the SDK (`pyAgxArm`) | **yes** |
| `arm/control.py` | rates, gains, gravity compensation, impedance | **yes** |
| `arm/fake.py` | offline stand-in for the arm | no |
| `robot.py` | URDF, forward kinematics, output paths | no |
| `visualization/` | MeshCat views | no |
| `agx_reference/` | vendored [gravity-compensation/impedance code](https://github.com/kehuanjack/agilex-arm-gravity-compensation) (branch `imp`) and URDF; keep it close to upstream | `piper/` yes |
| `notebooks/Sequential_E2T2_peg_slide_2D.ipynb` | 2-D source of the method; `method/test_against_notebook.py` checks `law.py` against it | no |

## History

The full pre-handoff state, including the experiment notes and the evaluation runs, is on
branch `archive/pre-handoff-20261119`. Tags `door-working-v1` and `door-experiment-v*` mark the
code of the door experiments.
