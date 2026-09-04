#!/bin/bash

export USR_NAME=jens
export IMAGE_NAME=docker-piper-ergodic
export CONTAINER_NAME=piper-ergodic


docker run --rm -it \
    --name ${CONTAINER_NAME} \
    --privileged \
    --net=host \
    --ipc=host \
    \
	--volume "/home/${USR_NAME}/Ergodic_Exploration_using_Tensor_Train:/home/${USR_NAME}/workspace/Ergodic_Exploration_using_Tensor_Train" \
	\
    --volume "/home/${USR_NAME}/docker_intern_PiPER_ergodic:/home/${USR_NAME}/workspace/docker_intern_PiPER_ergodic" \
    \
    ${IMAGE_NAME} \
	\
    bash
