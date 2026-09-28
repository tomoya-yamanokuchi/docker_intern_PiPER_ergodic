#!/bin/bash

export USR_NAME=jens
export IMAGE_NAME=docker-piper-ergodic
export CONTAINER_NAME=piper-ergodic

# X11 cookie for the container. meshcat needs none of this -- it renders in the
# host browser -- but matplotlib windows do, if anything here opens one.
#
# The host's own cookie lives under /run/user/1004/gdm/, which is awkward to
# mount, so it is copied out here. The sed rewrites the address family to
# FamilyWild, which is what makes a cookie issued for this host valid from
# inside the container. Re-running is harmless.
#
# Note the host display is :1, not :0, and the image already carries python3-tk,
# so no rebuild is needed for this.
export XAUTH=/tmp/.docker.xauth

touch ${XAUTH}
xauth nlist "${DISPLAY}" | sed -e 's/^..../ffff/' | xauth -f ${XAUTH} nmerge -
chmod 644 ${XAUTH}

# PYTHONPATH carries two import roots: our code in workspace/src, and
# agx_reference, whose files import core.* / controller.* as top-level packages.
#
# OPENBLAS_NUM_THREADS=1 because OpenBLAS spawns threads across all 24 cores even
# for the tiny SVDs of ErgodicController.step's tensor-train rounding, and the spawn
# dominates: one round(rmax=36) costs up to 118 ms multithreaded against 2 ms on one
# thread, which overran the 100 Hz control loop by 30x. Nothing here has a matrix
# large enough to gain from threading -- the coefficient build is 20x faster too. It
# belongs in the environment because OpenBLAS reads it when it loads, before any
# Python could set it.
docker run --rm -it \
    --name ${CONTAINER_NAME} \
    --privileged \
    --net=host \
    --ipc=host \
    \
    --env DISPLAY="${DISPLAY}" \
    --env XAUTHORITY=${XAUTH} \
    --volume "${XAUTH}:${XAUTH}:ro" \
    --volume "/tmp/.X11-unix:/tmp/.X11-unix:ro" \
    \
    --env PYTHONPATH="/home/${USR_NAME}/workspace/docker_intern_PiPER_ergodic/workspace/src:/home/${USR_NAME}/workspace/docker_intern_PiPER_ergodic/workspace/src/agx_reference" \
    \
    --env OPENBLAS_NUM_THREADS=1 \
    \
	--volume "/home/${USR_NAME}/Ergodic_Exploration_using_Tensor_Train:/home/${USR_NAME}/workspace/Ergodic_Exploration_using_Tensor_Train" \
	\
	--volume "/home/${USR_NAME}/agilex-arm-gravity-compensation:/home/${USR_NAME}/workspace/agilex-arm-gravity-compensation" \
	\
    --volume "/home/${USR_NAME}/docker_intern_PiPER_ergodic:/home/${USR_NAME}/workspace/docker_intern_PiPER_ergodic" \
    \
    ${IMAGE_NAME} \
	\
    bash
