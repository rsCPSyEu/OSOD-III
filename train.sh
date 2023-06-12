#!/bin/bash

python tools/train_net.py \
--num-gpus 1 \
--config-file configs/CUB200/random/faster_rcnn_R_50_FPN_3x_CUB200_t1.yaml
OUTPUT_DIR ./output/trial

# python tools/train_net.py \
# --num-gpus 1 \
# --config-file configs/CUB200/random/opendet_R_50_FPN_3x_CUB200_t1.yaml \
# --eval-only \
# MODEL.WEIGHTS ~/src/opendet2/output/CUB200/t1/model_final.pth

# python tools/train_net.py \
# --num-gpus 1 \
# --config-file configs/CUB200/random/faster_rcnn_R_50_FPN_3x_CUB200_t1.yaml \
# --eval-only \
# MODEL.WEIGHTS 