import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator

from visualizer import plot_losses, show_results
from hooks import ValidationLoss
from kitty_dataset import register_kitti_dataset, get_kitti_dicts


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
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Load Best Model For inference
    trainer.model.load_state_dict(val_loss.weights)
    trainer.test(cfg, trainer.model, evaluators=[evaluator])

    predictor = DefaultPredictor(cfg)
    predictor.model.load_state_dict(trainer.model.state_dict())

    dataset_dicts = get_kitti_dicts("data/KITTI/data_object_image_2/training/image_2/",  "data/KITTI/training/label_2/", is_train=False)

    plot_losses(cfg)
    show_results(cfg, dataset_dicts, predictor, samples=10)
