# PiPER Ergodic Control Environment

This repository provides a lightweight Docker environment for developing **ergodic control on the AgileX PiPER robot**.

The environment is intentionally kept simple:

- Ubuntu 22.04
- Python 3.10
- PiPER control through `piper_sdk` (no ROS)
- Tensor Train computations through the current `ttpy`
- E2T2: *Ergodic Exploration using Tensor Train*
- Optional JAX support for the JAX-based E2T2 notebook
- No ROS / RViz / Open3D

The main goal is to first reproduce:

1. direct PiPER control using `piper_sdk`, and
2. the original E2T2 ergodic-control examples,

before connecting the two.

---

## 1. Environment overview

```text
E2T2 / ergodic controller
        |
        |  Python 3.10
        v
     piper_sdk
        |
     python-can
        |
       can0
        |
      PiPER
```

ROS is **not** required for the experiments in this repository.

The current Tensor Train package is installed directly from the latest `ttpy` GitHub repository rather than using the old PyPI `ttpy 1.x` package.

---

## 2. Build the Docker image

From the root directory of this repository:

```bash
sh build.sh
```

The Docker image used in the current setup is:

```text
docker-piper-ergodic
```

A successful build should include checks similar to:

```text
Python 3.10.x
python-can: OK
piper_sdk.Piper: OK
ttpy: OK
```

---

## 3. Start the Docker container


Then start the container:

```bash
sh run.sh
```


## 4. Verify the Python environment

Inside Docker:

```bash
python --version
```

Expected:

```text
Python 3.10.x
```

Check PiPER SDK:

```bash
python -c "from piper_sdk import Piper; print('Piper OK')"
```

Check Tensor Train:

```bash
python -c "import tt; print('TT OK')"
```

Check `python-can`:

```bash
python -c "import can; print('CAN OK')"
```

---

# Part A: Test PiPER control

## 5. Check the CAN interface

Before running any robot-control script:

```bash
ip link show can0
```

A working interface should look similar to:

```text
can0: <NOARP,UP,LOWER_UP,...>
```

For more detail:

```bash
ip -details link show can0
```

The PiPER CAN bitrate is normally:

```text
1000000
```



## 6. Run the existing PiPER demo


Go to:

```bash
cd /home/jens/workspace/docker_PiPER_env_ver2/double_PiPER/src/test
```


Then run:

```bash
python test_ctrlPiperJoint_can0.py
```

This has been confirmed to work with:

```text
Ubuntu 22.04
Python 3.10
piper_sdk
python-can
```

No ROS node is required.

---

## Safety note for PiPER

`test_ctrlPiperJoint_can0.py` sends real commands to the robot.

Before running it:

- make sure the robot is in a safe configuration,

The existing demo also contains a safety sequence for switching from Teaching mode to CAN command mode. Do not remove the `stop()` / `enable()` logic when testing real hardware.

---

# Part B: Run E2T2

The E2T2 implementation used here is:

```text
SuhanNShetty/Ergodic_Exploration_using_Tensor_Train
```

Clone it inside the mounted workspace if it is not already present:

```bash
cd /home/jens/workspace

git clone https://github.com/SuhanNShetty/Ergodic_Exploration_using_Tensor_Train.git

cd Ergodic_Exploration_using_Tensor_Train
```

The repository contains:

```text
Ergodic_Exploration_using_TT_python310.ipynb
Ergodic_Exploration_using_TT_with_JAX_python310.ipynb
README.md
```

---

## 7. Start JupyterLab

From the E2T2 repository:

```bash
jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --allow-root \
    --no-browser
```

Because Docker uses:

```text
--net=host
```

open the URL printed by Jupyter in the host browser, typically:

```text
http://127.0.0.1:8888/lab?token=...
```

---

# Part C: Standard E2T2 notebook

Open:

```text
Ergodic_Exploration_using_TT_python310.ipynb
```


# Part D: JAX-based E2T2 notebook

Open:

```text
Ergodic_Exploration_using_TT_with_JAX_python310.ipynb
```

The JAX version mainly accelerates the **pre-processing step**, especially the computations used to obtain the Fourier coefficients.

---

## 13. Install JAX

The tested versions for this Python 3.10 environment are:

```bash
python -m pip install "jax==0.5.3" "jaxlib==0.5.3"
```

For long-term reproducibility, it is recommended to add these pinned versions to the Dockerfile and rebuild the image.

After installation, restart the Jupyter kernel.


# Part F: Next development steps

....


## References

- E2T2: `SuhanNShetty/Ergodic_Exploration_using_Tensor_Train`
- Tensor Train toolbox: `oseledets/ttpy`
- PiPER SDK: `agilexrobotics/piper_sdk`
- Original PiPER Docker/test setup: `Issa-N/docker_PiPER_env_ver2`
