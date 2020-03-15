import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer
import random
from detectron2.data import MetadataCatalog
import cv2

from visualizer import plot_losses, show_results
from hooks import ValidationLoss
from kitti_mots_dataset import register_kitti_mots_dataset, get_kiti_mots_dicts


if __name__ == '__main__':


    n_imgs_train = 37664
    batch_size = 4
    epoch = n_imgs_train // batch_size + 1 #  1 epoch

    register_kitti_mots_dataset("datasets/KITTI-MOTS/training/image_02",
                                "datasets/KITTI-MOTS/instances_txt",
                                ("kitti_mots_train", "kitti_mots_test"),
                                image_extension="png")
    register_kitti_mots_dataset("datasets/MOTSChallenge/train/images",
                                "datasets/MOTSChallenge/train/instances_txt",
                                ("mots_challenge_train", "mots_challenge_test"),
                                image_extension="jpg")

    # DATA
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    # cfg.DATASETS.TRAIN = ("kitti_mots_train", "mots_challenge_train", )
    cfg.DATASETS.TRAIN = ("kitti_mots_train", )
    cfg.DATASETS.TEST = ("kitti_mots_test", )
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo

    # PARAMETERS
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = 0.0002 * batch_size / 16 # pick a good LR
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    # cfg.SOLVER.MAX_ITER = 200
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model

    # LOOP
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    evaluator = COCOEvaluator("kitti_mots_test", cfg, False, output_dir="output/")
    # val_loss = ValidationLoss(cfg)
    # trainer.register_hooks([val_loss])
    # trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    # trainer.resume_or_load(resume=False)
    # trainer.train()

    # Load Best Model For inference
    # trainer.model.load_state_dict(val_loss.weights)
    trainer.test(cfg, trainer.model, evaluators=[evaluator])

    predictor = DefaultPredictor(cfg)
    # predictor.model.load_state_dict(trainer.model.state_dict())

    dataset_dicts = get_kiti_mots_dicts("datasets/KITTI-MOTS/training/image_02",  "datasets/KITTI-MOTS/instances_txt", is_train=False, image_extension='png')

    plot_losses(cfg)
    show_results(cfg, dataset_dicts, predictor, samples=10)
