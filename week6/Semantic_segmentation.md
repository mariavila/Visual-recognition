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

### Experiment 1
Training:
```
python3 deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=90000 \
    --train_split="train_fine" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --train_crop_size="769,769" \
    --train_batch_size=1 \
    --add_image_level_feature=True \
    --dataset="cityscapes" \
    --fine_tune_batch_norm=False \
    --tf_initial_checkpoint=/path/to/imagenet/checkpoint \
    --train_logdir=/path/to/log/dir \
    --dataset_dir=/path/to/cityscapes/tfrecord/dir
```
Evaluation:
```
python deeplab/eval.py \
    --logtostderr \
    --eval_split="val_fine" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --eval_crop_size="1025,2049" \
    --add_image_level_feature=True \
    --dataset="cityscapes" \
    --checkpoint_dir=/path/to/trained/weights \
    --eval_logdir=/path/to/eval/log/dir \
    --dataset_dir=/path/to/cityscapes/tfrecord/dir
```
Visualization
```
python3 deeplab/vis.py \
    --logtostderr \
    --vis_split="val_fine" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --vis_crop_size="1025,2049" \
    --dataset="cityscapes" \
    --add_image_level_feature=True \
    --colormap_type="cityscapes" \
    --checkpoint_dir=/path/to/trained/weights \
    --vis_logdir=/path/to/vis/log/dir \
    --dataset_dir=/path/to/cityscapes/tfrecord/dir
```
### Experiment 2
Training:
```
python3 deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=90000 \
    --train_split="train_fine" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size="769,769" \
    --train_batch_size=4 \
    --add_image_level_feature=True \
    --dataset="cityscapes" \
    --fine_tune_batch_norm=False \
    --tf_initial_checkpoint=/path/to/imagenet/checkpoint \
    --train_logdir=/path/to/log/dir \
    --dataset_dir=/path/to/cityscapes/tfrecord/dir
```
Evaluation:
```
python deeplab/eval.py \
    --logtostderr \
    --eval_split="val_fine" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size="1025,2049" \
    --add_image_level_feature=True \
    --dataset="cityscapes" \
    --checkpoint_dir=/path/to/trained/weights \
    --eval_logdir=/path/to/eval/log/dir \
    --dataset_dir=/path/to/cityscapes/tfrecord/dir

```
Visualization:
```
python3 deeplab/vis.py \
    --logtostderr \
    --vis_split="val_fine" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size="1025,2049" \
    --dataset="cityscapes" \
    --add_image_level_feature=True \
    --colormap_type="cityscapes" \
    --checkpoint_dir=/path/to/trained/weights \
    --vis_logdir=/path/to/vis/log/dir \
    --dataset_dir=/path/to/cityscapes/tfrecord/dir
```

### Experiment 3
Training:
```
python3 deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=90000 \
    --train_split="train_fine" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size="769,769" \
    --train_batch_size=2 \
    --add_image_level_feature=False \
    --dataset="cityscapes" \
    --fine_tune_batch_norm=False \
    --tf_initial_checkpoint=/path/to/imagenet/checkpoint \
    --train_logdir=/path/to/log/dir \
    --dataset_dir=/path/to/cityscapes/tfrecord/dir
```
Evaluation:
```
python deeplab/eval.py \
    --logtostderr \
    --eval_split="val_fine" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size="1025,2049" \
    --add_image_level_feature=False \
    --dataset="cityscapes" \
    --checkpoint_dir=/path/to/trained/weights \
    --eval_logdir=/path/to/eval/log/dir \
    --dataset_dir=/path/to/cityscapes/tfrecord/dir
```
Visualization:
```
python3 deeplab/vis.py \
    --logtostderr \
    --vis_split="val_fine" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size="1025,2049" \
    --dataset="cityscapes" \
    --add_image_level_feature=False \
    --colormap_type="cityscapes" \
    --checkpoint_dir=/path/to/trained/weights \
    --vis_logdir=/path/to/vis/log/dir \
    --dataset_dir=/path/to/cityscapes/tfrecord/dir
```

### Experiment 4
Training:
```
python3 deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=90000 \
    --train_split="train_fine" \
    --model_variant="xception_71l" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size="769,769" \
    --train_batch_size=2 \
    --add_image_level_feature=False \
    --dataset="cityscapes" \
    --fine_tune_batch_norm=False \
    --tf_initial_checkpoint=/path/to/imagenet/checkpoint \
    --train_logdir=/path/to/log/dir \
    --dataset_dir=/path/to/cityscapes/tfrecord/dir
```
Evaluation:
```
python deeplab/eval.py \
    --logtostderr \
    --eval_split="val_fine" \
    --model_variant="xception_71" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size="1025,2049" \
    --add_image_level_feature=False \
    --dataset="cityscapes" \
    --checkpoint_dir=/path/to/trained/weights \
    --eval_logdir=/path/to/eval/log/dir \
    --dataset_dir=/path/to/cityscapes/tfrecord/dir
```
Visualization:
```
python3 deeplab/vis.py \
    --logtostderr \
    --vis_split="val_fine" \
    --model_variant="xception_71" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size="1025,2049" \
    --dataset="cityscapes" \
    --add_image_level_feature=False \
    --colormap_type="cityscapes" \
    --checkpoint_dir=/path/to/trained/weights \
    --vis_logdir=/path/to/vis/log/dir \
    --dataset_dir=/path/to/cityscapes/tfrecord/dir
```
