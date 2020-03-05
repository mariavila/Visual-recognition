import os
from glob import glob

import cv2
import numpy as np
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

classes_correspondence = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': 8
}


def get_annot(line):
    l = line.split()
    box = [float(l[4]), float(l[5]), float(l[6]), float(l[7])]
    cat_id = classes_correspondence[l[0]]
    return cat_id, box


def get_kitti_dicts(ims_path, annots_path, is_train=False, percentage_training=0.7):
    assert os.path.exists(ims_path)
    assert os.path.exists(annots_path)

    ims_files = glob(os.path.join(ims_path, "*.png"))
    ims_files = np.array(ims_files)
    ims_ids = np.arange(len(ims_files))

    max_train_ims = int(ims_ids.shape[0] * percentage_training)
    train_ids = ims_ids[:max_train_ims]
    test_ids = ims_ids[max_train_ims:]

    if is_train:
        ims = ims_files[train_ids]
    else:
        ims = ims_files[test_ids]

    kitti_dataset = []

    for i, im in enumerate(ims):
        h, w = cv2.imread(im).shape[:2]
        im_name = os.path.basename(im).split(".")[0]

        annot_file = os.path.join(annots_path, "{}.txt".format(im_name))
        assert os.path.exists(annot_file)

        im_annots = []
        with open(annot_file) as f:
            for annot in f.readlines():
                cat_id, box = get_annot(annot)

                annot = {
                    "category_id": cat_id,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "bbox": box
                }
                im_annots.append(annot)

        im_data = {
            "file_name": im,
            "image_id": i,
            "height": h,
            "width": w,
            "annotations": im_annots
        }
        kitti_dataset.append(im_data)

    return kitti_dataset


def register_kitti_dataset(ims_path, annots_path, train_percent=0.7):
    kitti_train = lambda: get_kitti_dicts(ims_path, annots_path, is_train=True, percentage_training=train_percent)
    kitti_test = lambda: get_kitti_dicts(ims_path, annots_path, is_train=False, percentage_training=train_percent)

    DatasetCatalog.register("kitti_train", kitti_train)
    MetadataCatalog.get("kitti_train").set(thing_classes=[k for k, v in classes_correspondence.items()])
    DatasetCatalog.register("kitti_test", kitti_test)
    MetadataCatalog.get("kitti_test").set(thing_classes=[k for k, v in classes_correspondence.items()])


if __name__ == '__main__':
    # np.random.seed = 50320

    register_kitti_dataset("data/KITTI/data_object_image_2/training/image_2/", "data/KITTI/training/label_2/")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("kitti_train",)
    cfg.DATASETS.TEST = ("kitti_test",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

