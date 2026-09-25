# PiPER Ergodic Control Environment

A Docker environment for developing **ergodic control on the AgileX PiPER
6-DoF arm**, without ROS. Ubuntu 22.04, Python 3.10, control over SocketCAN.

```
E2T2 / ergodic controller  ->  Python 3.10  ->  pyAgxArm
  ->  python-can  ->  socketcan  ->  can0 @ 1 Mbit/s  ->  PiPER
```

## Where the working code is

`workspace/src/agx_reference/` is the current working implementation: a
**Cartesian impedance** controller, a **joint impedance** controller and a
**gravity-compensation** demo, all validated on the arm. It is a verbatim copy of
[`kehuanjack/agilex-arm-gravity-compensation`](https://github.com/kehuanjack/agilex-arm-gravity-compensation)
@ `imp`, kept byte-identical to the version that was validated here, and it
carries its own URDF and meshes.

```
workspace/src/
└── agx_reference/     # the working impedance + gravity-compensation controllers
    ├── core/          # Pinocchio wrapper: FK, Jacobian, dynamics   (no SDK)
    ├── controller/    # Cartesian and joint impedance torque laws   (no SDK)
    └── piper/         # the three control loops -- these talk to the arm
```

That is the whole tree. `CLAUDE.md` is the only documentation.

`double_PiPER/` is the vendor's ROS1 package tree, kept only as reference. No
ROS is used anywhere here.

## Running it

Bring up the CAN interface once per boot or USB re-plug — the gs_usb adapter
comes up DOWN with no bitrate, and the PiPER always expects 1 Mbit/s:

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

Start the container, then:

```bash
cd /home/jens/workspace/docker_intern_PiPER_ergodic/workspace/src/agx_reference

python piper/main_gc.py                # gravity compensation
python piper/main_jnt_imp.py           # joint impedance hold
python piper/main_tast_imp.py          # Cartesian impedance hold
```

All three control demos run until Ctrl-C, which hands each joint to a position
hold so the arm stays up after Python exits.

> ⚠️ **The arm has no holding brakes.** It falls the instant it loses motor
> torque — cutting AC, the E-stop, disabling the arm, or a MIT command with all
> gains and the feedforward torque at zero. Support it before any deliberate
> power-down.

`CLAUDE.md` has the full detail: hardware safety, CAN bring-up and
troubleshooting, the `pyAgxArm` API, and the controller's validated gains.
