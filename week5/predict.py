import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from kitti_mots_dataset import get_kiti_mots_dicts
from tqdm import tqdm
from detectron2.engine import DefaultPredictor

from week5.visualizer import show_results


def filter_preds(preds, mapping_pretrain_to_dataset, mots):
    for pred in tqdm(preds, desc="Filtering predictions"):
        pred['instances'] = [i for i in pred['instances'] if i['category_id'] in mapping_pretrain_to_dataset.keys()]
        for instance in pred['instances']:
            instance['category_id'] = 0 if mots else mapping_pretrain_to_dataset[instance['category_id']]

    return preds


def inference(config_file, correspondences):
    # test_set = 'kitti_mots_test'
    test_set = 'mots_challenge_train'

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    cfg.DATASETS.TRAIN = (test_set,)
    cfg.DATASETS.TEST = (test_set,)
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    evaluator = COCOEvaluator(test_set, cfg, False, output_dir="./output/")
    print(evaluator._metadata.get("thing_classes"))
    val_loader = build_detection_test_loader(cfg, test_set)
    inference_on_dataset(trainer.model, val_loader, evaluator)

    preds = evaluator._predictions

    filtered_preds = filter_preds(preds, correspondences)
    evaluator._predictions = filtered_preds

    evaluator.evaluate()

    predictor = DefaultPredictor(cfg)
    motschallenge = DatasetCatalog.get(test_set)
    show_results(cfg, motschallenge, predictor)


if __name__ == '__main__':
    from week5.kitti_mots_dataset import kitti_correspondences, mots_correspondences
    ims_path_kitti = "../datasets/KITTI-MOTS/training/image_02"
    annots_path_kitti = "../datasets/KITTI-MOTS/instances_txt"
    ims_path = "../datasets/MOTSChallenge/MOTSChallenge/train/images"
    annots_path = "../datasets/MOTSChallenge/MOTSChallenge/train/instances_txt"
    train_percent_kitti_mots = 0.75


    def kitti_mots_train(): return get_kiti_mots_dicts(ims_path_kitti, annots_path_kitti, is_train=True,
                                                       train_percentage=train_percent_kitti_mots, image_extension='png')


    def kitti_mots_test(): return get_kiti_mots_dicts(ims_path_kitti, annots_path_kitti, is_train=False,
                                                      train_percentage=train_percent_kitti_mots, image_extension='png')

    def mots_challenge_train(): return get_kiti_mots_dicts(ims_path, annots_path, is_train=True, train_percentage=1.,
                                                           image_extension='jpg')


    DatasetCatalog.register("kitti_mots_train", kitti_mots_train)
    MetadataCatalog.get("kitti_mots_train").set(thing_classes=[k for k, v in kitti_correspondences.items()])

    DatasetCatalog.register("kitti_mots_test", kitti_mots_test)
    MetadataCatalog.get("kitti_mots_test").set(thing_classes=[k for k, v in kitti_correspondences.items()])

    DatasetCatalog.register("mots_challenge_train", mots_challenge_train)
    MetadataCatalog.get("mots_challenge_train").set(thing_classes=[k for k, v in mots_correspondences.items()])

    #        | COCO | KITTI | MOTSChallenge
    # person | 0    | 1     | 1
    # Car    | 2    | 0     | -

    # dict that maps category_id from coco dataset to the corresponding ones in kitti
    coco_to_kitti_dict = {
        2: 0,
        0: 1,
    }

    cityscapes_to_kiti_dict = {
        0: 1,
        2: 0
    }

    coco_to_mots_dict = {
        0: 1
    }

    inference("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml", coco_to_mots_dict)
    inference("Cityscapes/mask_rcnn_R_50_FPN.yaml", coco_to_mots_dict)


