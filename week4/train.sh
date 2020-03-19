#!/usr/bin/env bash

# python train_net.py COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml --output output/r50_dc5 --train_only # 1
# python train_net.py COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml --output output/r101_dc5 --train_only # 2
CUDA_VISIBLE_DEVICES=1 python train_net.py COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml --output output/r101_c4 --train_only # 3
# python train_net.py COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml --output output/r101_fpn --train_only # 4
python train_net.py COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --output output/r50_fpn --train_only # 5
python train_net.py Cityscapes/mask_rcnn_R_50_FPN.yaml --output output/r50_fpn_cityscapes --train_only # 6


# Compare 1 and 2 for depth
# Compare 2 and 3 for backbone configuration
# Compare 2 and 3 and 4 for FPN improvement
# Compare 5 and 6 for training data
