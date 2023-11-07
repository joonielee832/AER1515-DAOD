#!/bin/bash

DRIVE=$1

VOLUMES="--volume=${PWD}/misc:/home/misc
        --volume=${PWD}/src:/home/src
        --volume=${PWD}/results:/home/results
        --volume=${DRIVE}:/home/data"

GPU='"device=0"'

docker run \
-it \
-p 6006:6006 \
--privileged \
-e DISPLAY=unix$DISPLAY \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
--gpus $GPU \
--shm-size 32G \
$VOLUMES \
--name=aer1515-project \
aer1515-project
