from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
import cv2
import random
import torch

from kitti_mots_dataset import get_kiti_mots_dicts, register_kitti_mots_dataset

# Task predict from pretrained model (uses COCO classes)
if __name__ == '__main__':

    # cfg_file = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    cfg_file = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_file)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    register_kitti_mots_dataset("datasets/KITTI-MOTS/training/image_02",
                                "datasets/KITTI-MOTS/instances_txt",
                                ("kitti_mots_train", "kitti_mots_test"),
                                image_extension="png")

    cfg.DATASETS.TRAIN = ("kitti_mots_train", )
    cfg.DATASETS.TEST = ("kitti_mots_test", )

    evaluator = COCOEvaluator("kitti_mots_test", cfg, False, output_dir="output/")
    trainer = DefaultTrainer(cfg)
    trainer.test(cfg, model, evaluators=[evaluator])
