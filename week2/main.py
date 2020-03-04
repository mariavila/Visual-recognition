import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def plot_preds(im, cfg, predictor):
    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=2.0)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("1", v.get_image()[:, :, ::-1])
    cv2.waitKey(0)


def setup(config, thr=0.5):

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thr  # set threshold for this model
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = thr
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config)
    predictor = DefaultPredictor(cfg)

    return cfg, predictor

def main():
    cfg, predictor = setup("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml", thr=0.5)

    im = cv2.imread("data/MIT_split/test/highway/art568.jpg")
    plot_preds(im, cfg, predictor)
    im = cv2.imread("data/MIT_split/test/coast/land340.jpg")
    plot_preds(im, cfg, predictor)
    im = cv2.imread("data/MIT_split/test/street/artc14.jpg")
    plot_preds(im, cfg, predictor)
    im = cv2.imread("data/MIT_split/test/coast/n286096.jpg")
    plot_preds(im, cfg, predictor)
    im = cv2.imread("data/MIT_split/test/coast/natu813.jpg")
    plot_preds(im, cfg, predictor)

if __name__ == "__main__":
    main()



