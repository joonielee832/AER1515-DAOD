#!/bin/bash

#? Train
python3 train_net.py \
    --num-gpus 1 \
    --config configs/faster_rcnn_VGG_multiweather_city.yaml \
    OUTPUT_DIR /home/results/multiweather_city

#? Test
# python3 train_net.py \
#     --eval-only \
#     --num-gpus 1 \
#     --config configs/faster_rcnn_VGG_multiweather_city.yaml \
#     MODEL.WEIGHTS /home/misc/City2FogModified.pth