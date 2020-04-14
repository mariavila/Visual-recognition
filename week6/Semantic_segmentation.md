# Semantic segmentation

In this section we include instructions on how to install and run the DeepLabv3+ framework available [here](https://github.com/tensorflow/models/tree/master/research/deeplab).

## Installation
1. Follow the oficial installation guide available [here](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md).
2. Download the Cityscapes dataset and store it under /research/deeplab/datasets/cityscapes
3. Clone [Cityscapescripts](https://github.com/mcordts/cityscapesScripts) in /research/deeplab/datasets/cityscapes
4. Install Cityscapescripts with: 
```
pip install cityscapescripts 
```
5. Execute /research/deeplab/datasets/convert_cityscapes.sh
6. Go to /research/deeplab/datasets/cityscapes/tfrecord:
    1. Change train-* for train_fine-*
    2. Change val-* for val_fine-*

## Usage
The four experiments have been executed with the following commands:
Experiment 1

Experiment 2

Experiment 3

Experiment 4
