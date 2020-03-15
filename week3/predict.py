from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import cv2
import random

from kitti_mots_dataset import get_kiti_mots_dicts
from visualizer import show_results

# Task predict from pretrained model (uses COCO classes)
if __name__ == '__main__':
    cfg_file = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_file)

    predictor = DefaultPredictor(cfg)

    mots_dict = get_kiti_mots_dicts("/home/devsodin/datasets/KITTI-MOTS/training/image_02",  "/home/devsodin/datasets/KITTI-MOTS/instances_txt", is_train=False, image_extension='png')
    show_results(cfg, mots_dict, predictor, samples=10)
