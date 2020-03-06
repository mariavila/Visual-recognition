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
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader

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
    def kitti_train(): return get_kitti_dicts(ims_path, annots_path,
                                              is_train=True, percentage_training=train_percent)
    def kitti_test(): return get_kitti_dicts(ims_path, annots_path,
                                             is_train=False, percentage_training=train_percent)

    DatasetCatalog.register("kitti_train", kitti_train)
    MetadataCatalog.get("kitti_train").set(
        thing_classes=[k for k, v in classes_correspondence.items()])
    DatasetCatalog.register("kitti_test", kitti_test)
    MetadataCatalog.get("kitti_test").set(
        thing_classes=[k for k, v in classes_correspondence.items()])
