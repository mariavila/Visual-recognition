import os
import random

import cv2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer
from hooks import ValidationLoss
from kitti_mots_dataset import register_kitti_mots_dataset
from visualizer import plot_losses


def train(output, iou=None, nms=None, rpn=None):
    batch_size = 2
    DatasetCatalog.clear()
    register_kitti_mots_dataset("datasets/KITTI-MOTS/training/image_02",
                                "datasets/KITTI-MOTS/instances_txt",
                                ("kitti_mots_train", "kitti_mots_test"),
                                image_extension="png")
    cfg_file = "Cityscapes/mask_rcnn_R_50_FPN.yaml"
    output_dir = output

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_file)
    cfg.SEED = 42

    cfg.DATASETS.TRAIN = ("kitti_mots_train",)
    cfg.DATASETS.TEST = ("kitti_mots_test",)
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = 0.0002 * batch_size / 16  # pick a good LR
    cfg.SOLVER.MAX_ITER = 7500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = output_dir

    if iou is not None:
        cfg.MODEL.RPN.IOU_THRESHOLDS = [iou[0], iou[1]]

    if nms is not None:
        cfg.MODEL.RPN.NMS_THRESH = nms

    if rpn is not None:
        cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = rpn[0]
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = rpn[1]

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=True)
    trainer.train()
    evaluator = COCOEvaluator("kitti_mots_test", cfg, False, output_dir=output_dir)
    trainer.test(cfg, trainer.model, evaluators=[evaluator])
    plot_losses(cfg)


def save_ims_from_model(output_dir):
    register_kitti_mots_dataset("datasets/KITTI-MOTS/training/image_02",
                                "datasets/KITTI-MOTS/instances_txt",
                                ("kitti_mots_train", "kitti_mots_test"),
                                image_extension="png")

    cfg = get_cfg()
    cfg_file = "Cityscapes/mask_rcnn_R_50_FPN.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
    cfg.SEED = 42

    cfg.MODEL.WEIGHTS = os.path.join(output_dir, "model_final.pth")
    cfg.DATASETS.TRAIN = ("kitti_mots_test", )
    cfg.DATASETS.TEST = ("kitti_mots_test",)
    predictor = DefaultPredictor(cfg)

    mots_test = DatasetCatalog.get("kitti_mots_test")

    random.seed(991289902352970059272393463766778531712405567416019953137427298712849420652624107943398234695660286007524334222721892010493181783407)

    for data in random.sample(mots_test, 10):
        im = cv2.imread(data["file_name"])
        basename = os.path.basename(data['file_name'])
        outputs = predictor(im)

        outputs["instances"] = outputs["instances"][torch.where(outputs["instances"].pred_classes < 2)]
        v = Visualizer(
            im[:, :, ::-1], MetadataCatalog.get("coco_2017_val"), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(output_dir, basename), v.get_image()[:, :, ::-1])

if __name__ == '__main__':
    # train(".output/iou/0109", iou=(0.1, 0.9))
    # train(".output/iou/0208", iou=(0.2, 0.8))
    # train(".output/iou/0406", iou=(0.4, 0.6))
    # train(".output/iou/0505", iou=(0.5, 0.5))
    #
    # train(".output/nms/05", nms=0.5)
    # train(".output/nms/06", nms=0.6)
    # train(".output/nms/08", nms=0.8)
    # train(".output/nms/09", nms=0.9)
    #
    # train(".output/proposals/6k", rpn=(6000, 3000))
    # train(".output/proposals/9k", rpn=(9000, 4500))

    save_ims_from_model(".output/iou/0109")
    save_ims_from_model(".output/iou/0208")
    save_ims_from_model(".output/iou/0406")
    save_ims_from_model(".output/iou/0505")

    save_ims_from_model(".output/nms/05")
    save_ims_from_model(".output/nms/06")
    save_ims_from_model(".output/nms/08")
    save_ims_from_model(".output/nms/09")

    save_ims_from_model(".output/proposals/6k")
    save_ims_from_model(".output/proposals/9k")




