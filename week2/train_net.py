import os
import random

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer

from kitty_dataset import register_kitti_dataset, get_kitti_dicts
import torch
import cv2

if __name__ == '__main__':


    n_imgs_train = 5236
    batch_size = 8
    epoch = n_imgs_train // batch_size + 1 #  1 epoch

    register_kitti_dataset("data/KITTI/data_object_image_2/training/image_2/", "data/KITTI/training/label_2/")

    # DATA
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("kitti_train",)
    cfg.DATASETS.TEST = ("kitti_test",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo

    # PARAMETERS
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9
    cfg.SOLVER.MAX_ITER = epoch * 12
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model

    # LOOP
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    evaluator = COCOEvaluator("kitti_test", cfg, False, output_dir="output/")
    trainer.resume_or_load(resume=False)
    trainer.train()
    trainer.test(cfg, trainer.model, evaluators=[evaluator])

    cfg.MODEL.WEIGHTS = "output/model_final.pth"
    predictor = DefaultPredictor(cfg)
    dataset_dicts = get_kitti_dicts("data/KITTI/data_object_image_2/training/image_2/",  "data/KITTI/training/label_2/", is_train=False)

    for d in random.sample(dataset_dicts, 10):

        im = cv2.imread(d["file_name"])
        outputs = predictor(im)

        v = Visualizer(
            im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=2.0)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("1", v.get_image()[:, :, ::-1])
        cv2.waitKey(0)
