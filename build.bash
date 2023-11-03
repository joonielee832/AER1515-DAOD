#!/bin/bash

DIR=$(dirname "$0")
DIR=${DIR%/}

docker build --no-cache \
    -t aer1515-project \
    -f $DIR/docker/Dockerfile \
    $DIR/docker