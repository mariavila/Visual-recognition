import os
from glob import glob

import cv2
import numpy as np
from detectron2.structures import BoxMode

classes_correspondence = {
    'Car': 1,
    'Van': 2,
    'Truck': 3,
    'Pedestrian': 4,
    'Person_sitting': 5,
    'Cyclist': 6,
    'Tram': 7,
    'Misc': 8,
    'DontCare': 9
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

    np.random.shuffle(ims_ids)

    max_train_ims = int(ims_ids.shape[0] * percentage_training)
    train_ids = ims_ids[:max_train_ims]
    test_ids = ims_ids[max_train_ims:]

    if is_train:
        ims = ims_files[train_ids]
    else:
        ims = ims_files[test_ids]

    kitti_dataset = []

    for i, im in enumerate(ims):
        h, w = cv2.imread(im).shape[:-1]
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
        print(im_data)

        kitti_dataset.append(im_data)

    return kitti_dataset


if __name__ == '__main__':
    np.random.seed = 50320
    from detectron2.data import DatasetCatalog

    kitti_train = lambda: get_kitti_dicts("/content/drive/My Drive/KITTI/mini_train",
                                          "/content/drive/My Drive/KITTI/training/label_2", is_train=True)
    kitti_test = lambda: get_kitti_dicts("/content/drive/My Drive/KITTI/mini_train",
                                         "/content/drive/My Drive/KITTI/training/label_2", is_train=False)

    DatasetCatalog.register("kitti_train", kitti_train)
    DatasetCatalog.register("kitti_test", kitti_test)
