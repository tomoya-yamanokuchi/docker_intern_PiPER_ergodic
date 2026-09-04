FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Host user on Jens's PC:
# uid=1004(jens) gid=1004(jens)
ARG USERNAME=jens
ARG USER_UID=1004
ARG USER_GID=1004

ENV TZ=Asia/Tokyo
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1


#####################################################
# Basic system settings
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
    can-utils \
    iproute2 \
    ethtool \
    net-tools \
    iputils-ping \
    && locale-gen en_US.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


#####################################################
# Create user
#####################################################

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd \
        --uid ${USER_UID} \
        --gid ${USER_GID} \
        --create-home \
        --shell /bin/bash \
        ${USERNAME}


#####################################################
# Workspace
#####################################################

RUN mkdir -p /home/${USERNAME}/workspace \
    && chown -R ${USER_UID}:${USER_GID} /home/${USERNAME}

WORKDIR /home/${USERNAME}/workspace


#####################################################
# Python environment
#####################################################

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

RUN python -m pip install \
    python-can

RUN python -m pip install \
    "git+https://github.com/agilexrobotics/piper_sdk.git@1_0_0_beta"


#####################################################
# Tensor Train / Ergodic Control
#####################################################

RUN python -m pip install \
    "ttpy[fast] @ git+https://github.com/oseledets/ttpy"


#####################################################
# Build-time sanity check
#####################################################

RUN python --version && \
    python -c "import can; print('python-can: OK')" && \
    python -c "from piper_sdk import Piper; print('piper_sdk.Piper: OK')" && \
    python -c "import tt; print('ttpy: OK')"


#####################################################
# Run container as the host-compatible user
#####################################################

USER ${USERNAME}

CMD ["/bin/bash"]
