import os
from glob import glob

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from pycocotools import coco

classes_correspondence = {
    'Car': 0,
    'Pedestrian': 1
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
    seq_id = int(txt_file.split("/")[-1].split(".")[0])
    mots_annots = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
        m_frames = max([int(l[0]) for l in lines])

        for frame in range(1, m_frames + 1):
            frame_lines = [l for l in lines if int(l[0]) == frame]
            frame_annots = []

            image_name = "{:06d}.{}".format(frame, image_extension)
            if frame_lines:
                h, w = int(frame_lines[0][3]), int(frame_lines[0][4])

                for line in frame_lines:
                    cat_id = (int(line[1]) // 1000) - 1

                    if cat_id not in classes_correspondence.values():
                        continue

                    h, w = int(line[3]), int(line[4])
                    instance = int(line[1]) % 1000
                    segm = {
                        "counts": line[-1].strip(),
                        "size": [h, w]
                    }

                    box = coco.maskUtils.toBbox(segm).tolist()

                    annot = {
                        "category_id": cat_id,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "bbox": box
                    }
                    frame_annots.append(annot)

                im_data = {
                    "file_name": os.path.join(images_path, image_name),
                    "image_id": int(frame + seq_id * 1e6),
                    "height": h,
                    "width": w,
                    "annotations": frame_annots
                }
                mots_annots.append(im_data)
            else:
                pass


    return mots_annots


def register_kitti_mots_dataset(ims_path, annots_path, dataset_names, train_percent=0.7, image_extension='jpg'):
    assert isinstance(dataset_names, tuple), "dataset names should be a tuple with two strings (for train and test) "

    def kitti_mots_train(): return get_kiti_mots_dicts(ims_path, annots_path, is_train=True,
                                                       train_percentage=train_percent, image_extension=image_extension)

    def kitti_mots_test(): return get_kiti_mots_dicts(ims_path, annots_path, is_train=False,
                                                      train_percentage=train_percent, image_extension=image_extension)

    DatasetCatalog.register(dataset_names[0], kitti_mots_train)
    MetadataCatalog.get(dataset_names[0]).set(
        thing_classes=[k for k, v in classes_correspondence.items()])
    DatasetCatalog.register(dataset_names[1], kitti_mots_test)
    MetadataCatalog.get(dataset_names[1]).set(
        thing_classes=[k for k, v in classes_correspondence.items()])


if __name__ == '__main__':
    register_kitti_mots_dataset("/home/devsodin/datasets/KITTI-MOTS/training/image_02",
                                "/home/devsodin/datasets/KITTI-MOTS/instances_txt",
                                ("kitti_mots_train", "kitti_mots_test"),
                                image_extension="png")
    register_kitti_mots_dataset("/home/devsodin/datasets/MOTSChallenge/train/images",
                                "/home/devsodin/datasets/MOTSChallenge/train/instances_txt",
                                ("mots_challenge_train", "mots_challenge_test"),
                                image_extension="jpg")
    print("regiseted")