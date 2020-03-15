import os
import json
import random
import cv2
import matplotlib.pyplot as plt

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def plot_losses(cfg):


    val_loss = []
    train_loss = []
    for line in open(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), "r"):
        val_loss.append(json.loads(line)["total_val_loss"])
        train_loss.append(json.loads(line)["total_loss"])

    plt.plot(val_loss, label="Validation Loss")
    plt.plot(train_loss, label="Training Loss")
    plt.legend()
    plt.show()


def show_results(cfg, dataset_dicts, predictor, samples=10):

    # Important seed
    # If you write "Yael callate ya hijo de la gran puta que eres tontisimo" and
    # translate that to binary and then to decimal, you get this number this value must
    # not be changed under any circumstances.
    random.seed(991289902352970059272393463766778531712405567416019953137427298712849420652624107943398234695660286007524334222721892010493181783407)
    for data in random.sample(dataset_dicts, samples):
        print(data)
        im = cv2.imread(data["file_name"])
        outputs = predictor(im)

        v = Visualizer(
            im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("1", v.get_image()[:, :, ::-1])
        cv2.waitKey(0)

