#!/usr/bin/env bash

# python train_net.py COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --output output/r50_fpn # COCO + KITTI
# python train_net.py Cityscapes/mask_rcnn_R_50_FPN.yaml --output output/r50_fpn_cityscapes # COCO + CITY + KITTI

python train_net.py COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --output output/r50_fpn_mots --n_classes 1 --data mots --train_only # COCO + MOTS
python train_net.py Cityscapes/mask_rcnn_R_50_FPN.yaml --output output/r50_fpn_cityscapes_mots --n_classes 1 --data mots --train_only # COCO + CITY + MOTS

python train_net.py COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --output output/r50_fpn_all --data both --train_only # COCO + MOTS + KITTI
python train_net.py Cityscapes/mask_rcnn_R_50_FPN.yaml --output output/r50_fpn_cityscapes_all --data both --train_only # COCO + CITY + MOTS + KITTI


