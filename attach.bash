#!/bin/bash

sudo bash -c "echo '' > $(docker inspect --format="{{.LogPath}}" aer1515-project)"
docker attach --detach-keys="ctrl-a" aer1515-project