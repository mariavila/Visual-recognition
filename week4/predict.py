import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from kitti_mots_dataset import register_kitti_mots_dataset


def filter_preds(preds, cat_mapping_coco_kitti):
    for pred in preds:
        pred['instances'] = [i for i in pred['instances'] if i['category_id'] in cat_mapping_coco_kitti.keys()]
        for instance in pred['instances']:
            instance['category_id'] = cat_mapping_coco_kitti[instance['category_id']]

    return preds


def inference(config_file, coco_to_kitti_dict):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    cfg.DATASETS.TRAIN = ("kitti_mots_train",)
    cfg.DATASETS.TEST = ("kitti_mots_test",)
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    evaluator = COCOEvaluator("kitti_mots_test", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "kitti_mots_test")
    inference_on_dataset(trainer.model, val_loader, evaluator)

    preds = evaluator._predictions

    filtered_preds = filter_preds(preds, coco_to_kitti_dict)
    evaluator._predictions = filtered_preds

    evaluator.evaluate()


if __name__ == '__main__':
    register_kitti_mots_dataset("../datasets/KITTI-MOTS/training/image_02",
                                "../datasets/KITTI-MOTS/instances_txt",
                                ("kitti_mots_train", "kitti_mots_test"),
                                image_extension="png")

    # dict that maps category_id from coco dataset to the corresponding ones in kitti
    coco_to_kitti_dict = {
        2: 0,
        0: 1,
    }
    inference("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", coco_to_kitti_dict)
