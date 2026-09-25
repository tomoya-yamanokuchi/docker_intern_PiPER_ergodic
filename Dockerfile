FROM ubuntu:22.04


#####################################################
# Build arguments
#####################################################

ARG DEBIAN_FRONTEND=noninteractive

# Match the host user:
# uid=1004(jens) gid=1004(jens)
ARG USERNAME=jens
ARG USER_UID=1004
ARG USER_GID=1004


#####################################################
# Environment
#####################################################

ENV TZ=Asia/Tokyo

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1


#####################################################
# Basic system packages
#####################################################

RUN apt-get update && apt-get install -y \
    tzdata \
    locales \
    sudo \
    git \
    curl \
    wget \
    vim \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-tk \
    python-is-python3 \
    liblapack-dev \
    can-utils \
    iproute2 \
    ethtool \
    net-tools \
    iputils-ping \
    && locale-gen en_US.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


#####################################################
# Python environment
#####################################################

# Ubuntu 22.04 uses Python 3.10 as the default Python 3.
# python-is-python3 makes `python` point to Python 3.

RUN python -m pip install --upgrade \
    pip \
    setuptools \
    wheel


#####################################################
# Scientific Python packages
#####################################################

RUN python -m pip install \
    numpy \
    scipy \
    matplotlib \
    tqdm \
    ipython \
    jupyterlab


#####################################################
# PiPER SDK
#####################################################

# test_ctrlPiperJoint_can0.py uses:
#
#     from piper_sdk import Piper
#
# The Piper high-level API is provided by the
# 1_0_0_beta branch.

RUN python -m pip install \
    python-can

RUN python -m pip install \
    "git+https://github.com/agilexrobotics/piper_sdk.git@1_0_0_beta"

# AgileX's newer SDK, installed alongside piper_sdk rather than
# replacing it. Both import fine; only one may hold can0 at a
# time. It is here because it scales the MIT t_ff argument to
# real N.m per firmware profile, which piper_sdk does not, and a
# Cartesian impedance controller puts its whole output through
# t_ff.

RUN python -m pip install \
    "git+https://github.com/agilexrobotics/pyAgxArm.git"


#####################################################
# Tensor Train / Ergodic Control
#####################################################

# Install ttpy directly from GitHub.
#
# [fast] enables the optional numba accelerated
# implementation.

RUN python -m pip install \
    "ttpy[fast] @ git+https://github.com/oseledets/ttpy"

#####################################################
# Pinocchio / IK Solver and Collision checking
#####################################################
RUN python -m pip install --no-cache-dir \
 "pin==4.1.0" \
  meshcat \
  robot_descriptions

#####################################################
# Build-time sanity check
#####################################################

RUN python --version && \
    python -c "import can; print('python-can: OK')" && \
    python -c "from piper_sdk import Piper; print('piper_sdk.Piper: OK')" && \
    python -c "from pyAgxArm import AgxArmFactory; print('pyAgxArm: OK')" && \
    python -c "import tt; print('ttpy: OK')" && \
    python -c "import pinocchio as pin, coal; print(pin.__version__)"


#####################################################
# Create non-root user
#####################################################

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd \
        --uid ${USER_UID} \
        --gid ${USER_GID} \
        --create-home \
        --shell /bin/bash \
        ${USERNAME}


#####################################################
# Allow sudo inside the container
#####################################################

# Useful for CAN interface configuration such as
# `ip link set can0 ...`.
#
# The container is already intended to be run with
# --privileged, so passwordless sudo is convenient
# for this development environment.

RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" \
        > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}


#####################################################
# Workspace
#####################################################

RUN mkdir -p /home/${USERNAME}/workspace \
    && chown -R ${USER_UID}:${USER_GID} \
        /home/${USERNAME}

WORKDIR /home/${USERNAME}/workspace


#####################################################
# PiPER SDK log permission
#####################################################

# piper_sdk creates log directories inside its
# installed package directory when imported.
#
# Since the package itself is installed as root,
# explicitly give the normal container user
# ownership of only the log directory.

RUN PIPER_SDK_DIR="$(python -c \
    'import importlib.util, os; \
     print(os.path.dirname(importlib.util.find_spec("piper_sdk").origin))')" \
    && mkdir -p "${PIPER_SDK_DIR}/log" \
    && chown -R ${USER_UID}:${USER_GID} \
        "${PIPER_SDK_DIR}/log"


#####################################################
# Run as normal user
#####################################################

USER ${USERNAME}

WORKDIR /home/${USERNAME}/workspace


#####################################################
# Default command
#####################################################

CMD ["/bin/bash"]
