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
from opt import parse_args
from week6.dataloader import TrainerDA

if __name__ == '__main__':

    opts = parse_args()
    batch_size = 2

    register_kitti_mots_dataset("../datasets/KITTI-MOTS/training/image_02",
                                "../datasets/KITTI-MOTS/instances_txt",
                                ("kitti_mots_train", "kitti_mots_test"),
                                image_extension="png")

    # register_kitti_mots_dataset("datasets/MOTSChallenge/train/images",
    #                             "datasets/MOTSChallenge/train/instances_txt",
    #                             ("mots_challenge_train", "mots_challenge_test"),
    #                             image_extension="jpg")

    cfg_file = opts.config
    output_dir = opts.output

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_file)

    if opts.data == "kitti":
        cfg.DATASETS.TRAIN = ("kitti_mots_train", )

    elif opts.data == "mots":
        cfg.DATASETS.TRAIN = ("mots_challenge_train", )

    elif opts.data == "both":
        cfg.DATASETS.TRAIN = ("kitti_mots_train", "mots_challenge_train", )

    cfg.DATASETS.TEST = ("kitti_mots_test", )
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = 0.0002 * batch_size / 16 # pick a good LR
    cfg.SOLVER.MAX_ITER = 5000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = opts.n_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).set(crop=opts.crop, hflip=opts.hflip, change_contrast=opts.contrast)


    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = TrainerDA(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

    trainer.resume_or_load(resume=True)
    trainer.train()

    if not opts.train_only:
        evaluator = COCOEvaluator("kitti_mots_test", cfg, False, output_dir=output_dir)
        trainer.test(cfg, trainer.model, evaluators=[evaluator])
        plot_losses(cfg)

        predictor = DefaultPredictor(cfg)
        predictor.model.load_state_dict(trainer.model.state_dict())

        dataset_dicts = get_kiti_mots_dicts("../datasets/KITTI-MOTS/training/image_02", "../datasets/KITTI-MOTS/instances_txt", is_train=False, image_extension='png')
        show_results(cfg, dataset_dicts, predictor, samples=10)
