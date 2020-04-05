import copy

import numpy as np
import torch
from detectron2.data import build_detection_train_loader, MetadataCatalog, build_detection_test_loader, DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer


class CustomMapper(DatasetMapper):

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.crop = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get('crop')
        self.hflip = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get('hflip')
        self.change_contrast = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get('change_contrast')
        self.min_size = cfg.INPUT.MIN_SIZE_TRAIN if is_train else cfg.INPUT.MIN_SIZE_TEST
        self.max_size = cfg.INPUT.MAX_SIZE_TRAIN if is_train else cfg.INPUT.MAX_SIZE_TEST
        self.sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING if is_train else "choice"
        self.crop_type = cfg.INPUT.CROP.TYPE
        self.crop_size = cfg.INPUT.CROP.SIZE


    def __call__(self, dataset_dict):

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        utils.check_image_size(dataset_dict, image)

        # Crop (default detectron2)
        if self.sample_style == "range":
            assert len(self.min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
                len(self.min_size)
            )

        transform_list = []

        # Transforms order
        # crop
        # resize
        # hflip
        # color_transforms

        crop_transform = None
        if self.crop and self.is_train:
            crop_transform = T.RandomCrop(self.crop_type, self.crop_size)
            transform_list.append(crop_transform)

        # Always do resize
        transform_list.append(T.ResizeShortestEdge(self.min_size, self.max_size, self.sample_style))

        if self.hflip and self.is_train:
            transform_list.append(T.RandomFlip())
        if self.change_contrast and self.is_train:
            transform_list.append(T.RandomContrast(0.9, 1.1))

        image, transforms = T.apply_transform_gens(transform_list, image)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        image_shape = image.shape[:2]  # h, w

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:

            annos = [utils.transform_instance_annotations(obj, transforms, image_shape, keypoint_hflip_indices=None) for
                     obj
                     in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0]

            instances = utils.annotations_to_instances(annos, image_shape)
            # Create a tight bounding box from masks, useful when image is cropped
            if crop_transform is not None and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict


class TrainerDA(DefaultTrainer):

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=CustomMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=CustomMapper(cfg, True))


if __name__ == '__main__':
    data_loader = build_detection_train_loader(cfg, mapper=CustomMapper(cfg))
