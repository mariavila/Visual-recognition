import copy

import numpy as np
import torch
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


def custom_mapper(dataset_dict, cfg, is_train=True, crop=False, hflip=False, change_contrast=False):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # USER: Write your own image loading if it's not from a file
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    utils.check_image_size(dataset_dict, image)

    # Crop (default detectron2)
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
            len(min_size)
        )

    transform_list = []

    # Transforms order
    # crop
    # resize
    # hflip
    # color_transforms

    crop_transform = None
    if crop and is_train:
        crop_transform = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
        transform_list.append(crop_transform)

    # Always do resize
    transform_list.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    if hflip and is_train:
        transform_list.append(T.RandomFlip())
    if change_contrast and is_train:
        transform_list.append(T.RandomContrast(0.9, 1.1))

    image, transforms = T.apply_transform_gens(transform_list, image)

    # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
    # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
    # Therefore it's important to use torch.Tensor.
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

    image_shape = image.shape[:2]  # h, w

    if not is_train:
        # USER: Modify this if you want to keep them for some reason.
        dataset_dict.pop("annotations", None)
        dataset_dict.pop("sem_seg_file_name", None)
        return dataset_dict

    if "annotations" in dataset_dict:

        annos = [utils.transform_instance_annotations(obj, transforms, image_shape, keypoint_hflip_indices=None) for obj
                 in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0]

        instances = utils.annotations_to_instances(annos, image_shape)
        # Create a tight bounding box from masks, useful when image is cropped
        if crop_transform is not None and instances.has("gt_masks"):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict


if __name__ == '__main__':
    data_loader = build_detection_train_loader(cfg, mapper=custom_mapper)
