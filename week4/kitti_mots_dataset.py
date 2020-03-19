import os
import cv2
from glob import glob

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from pycocotools import coco
import numpy as np

classes_correspondence = {
    'Car': 0,
    'Pedestrian': 1,
}

# mots cat_id --> coco cat_id # esto no tira
coco_correspondence = {
    0: 2,
    1: 0,
}

def get_kiti_mots_dicts(images_folder, annots_folder, is_train, train_percentage=0.75, image_extension="jpg"):
    assert os.path.exists(images_folder)
    assert os.path.exists(annots_folder)

    annot_files = sorted(glob(os.path.join(annots_folder, "*.txt")))

    n_train_seqences = int(len(annot_files) * train_percentage)
    train_sequences = annot_files[:n_train_seqences]
    test_sequences = annot_files[n_train_seqences:]

    sequences = train_sequences if is_train else test_sequences

    kitti_mots_annotations = []
    for seq_file in sequences:
        seq_images_path = os.path.join(images_folder, seq_file.split("/")[-1].split(".")[0])
        kitti_mots_annotations += mots_annots_to_coco(seq_images_path, seq_file, image_extension)

    return kitti_mots_annotations


def mots_annots_to_coco(images_path, txt_file, image_extension):
    assert os.path.exists(txt_file)
    n_seq = int(txt_file.split("/")[-1].split(".")[0])

    mots_annots = []
    with open(txt_file, 'r') as f:
        annots = f.readlines()
        annots = [l.split() for l in annots]

        annots = np.array(annots)

        for frame in np.unique(annots[:, 0].astype('uint8')):

            frame_lines = annots[annots[:, 0] == str(frame)]
            if frame_lines.size > 0:

                h, w = int(frame_lines[0][3]), int(frame_lines[0][4])

                f_objs = []
                for a in frame_lines:
                    cat_id = int(a[2]) - 1
                    if cat_id in classes_correspondence.values():
                        # cat_id = coco_correspondence[cat_id]
                        segm = {
                            "counts": a[-1].strip().encode(encoding='UTF-8'),
                            "size": [h, w]
                        }

                        box = coco.maskUtils.toBbox(segm)
                        box[2:] = box[2:] + box[:2]
                        box = box.tolist()

                        # mask to poly
                        mask = np.ascontiguousarray(coco.maskUtils.decode(segm))
                        _, contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_TREE,
                                                        cv2.CHAIN_APPROX_SIMPLE)
                        poly = []

                        for contour in contours:
                            contour = contour.flatten().tolist()
                            if len(contour) > 4:
                                poly.append(contour)
                        if len(poly) == 0:
                            continue

                        annot = {
                            "category_id": cat_id,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "bbox": box,
                            "segmentation": poly

                        }
                        f_objs.append(annot)


                frame_data = {
                    "file_name": os.path.join(images_path, '{:06d}.{}'.format(int(a[0]), image_extension)),
                    "image_id": int(frame + n_seq * 1e6),
                    "height": h,
                    "width": w,
                    "annotations": f_objs
                }
                mots_annots.append(frame_data)

    return mots_annots


def register_kitti_mots_dataset(ims_path, annots_path, dataset_names, train_percent=0.75, image_extension='jpg'):
    assert isinstance(dataset_names, tuple), "dataset names should be a tuple with two strings (for train and test) "

    def kitti_mots_train(): return get_kiti_mots_dicts(ims_path, annots_path, is_train=True,
                                                       train_percentage=train_percent, image_extension=image_extension)

    def kitti_mots_test(): return get_kiti_mots_dicts(ims_path, annots_path, is_train=False,
                                                      train_percentage=train_percent, image_extension=image_extension)

    DatasetCatalog.register(dataset_names[0], kitti_mots_train)
    MetadataCatalog.get(dataset_names[0]).set(thing_classes=[k for k, v in classes_correspondence.items()])
    DatasetCatalog.register(dataset_names[1], kitti_mots_test)
    MetadataCatalog.get(dataset_names[1]).set(thing_classes=[k for k, v in classes_correspondence.items()])


if __name__ == '__main__':
    get_kiti_mots_dicts("/home/devsodin/datasets/KITTI-MOTS/training/image_02",
                                "/home/devsodin/datasets/KITTI-MOTS/instances_txt",
                                ("kitti_mots_train", "kitti_mots_test"),
                                image_extension="png")
    get_kiti_mots_dicts("/home/devsodin/datasets/MOTSChallenge/train/images",
                                "/home/devsodin/datasets/MOTSChallenge/train/instances_txt",
                                ("mots_challenge_train", "mots_challenge_test"),
                                image_extension="jpg")

    register_kitti_mots_dataset("/home/devsodin/datasets/KITTI-MOTS/training/image_02",
                                "/home/devsodin/datasets/KITTI-MOTS/instances_txt",
                                ("kitti_mots_train", "kitti_mots_test"),
                                image_extension="png")

    register_kitti_mots_dataset("/home/devsodin/datasets/MOTSChallenge/train/images",
                                "/home/devsodin/datasets/MOTSChallenge/train/instances_txt",
                                ("mots_challenge_train", "mots_challenge_test"),
                                image_extension="jpg")
    print("regiseted")
