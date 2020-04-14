# Visual-recognition

Project of the Visual Recognition course of the Computer Vision Master

Professors: [Carles Ventura](https://github.com/carlesventura) and [Jose Luis Gomez](https://github.com/JoseLGomez)

Team: Group 2

## Documents 📋
- Overleaf document for the [report](https://www.overleaf.com/read/mtngzprrpcsh)
- Overleaf document for the [FINAL REPORT](https://www.overleaf.com/read/xyqfntwpygfm)
- Slides summarizing the results:
  - [Week1](https://docs.google.com/presentation/d/1XOinqBwgxyKsabA3UqhsSe8kFd7tmZ2fbFGJMYC_Bvc/edit?usp=sharing)
  - [Week2](https://docs.google.com/presentation/d/1V4aaBV6_ox5YCAfNQBxpx8ERtwlDSt0yL3eFkl-cBII/edit?usp=sharing)
  - [Week3](https://docs.google.com/presentation/d/17GRGgdpLqLFxgFKv1ACBYeYmAGrZ891nKR1rK6gBnaY/edit?usp=sharing)
  - [Week4](https://docs.google.com/presentation/d/13_ZGLmGhX3iJQJ1d5xIs1IOq-L-4PpB2KARs_m5LWXM/edit?usp=sharing)
  - [Week5](https://docs.google.com/presentation/d/1PH0-71IMaPoaUZLSuIhORXeCeMNAWKZjdwqqDTsisOY/edit?usp=sharing)
  - [Week6](https://docs.google.com/presentation/d/1Sti804trqcxeBIJYKIdKyChygcdJLi2FMHxDDCdpIhc/edit?usp=sharing)
  
  
## Tasks progress 📈
### Mini-project
The mini-project consists on implementing in Pytorch the final classification network from M3 in order to get used to Pytorch.
* [x] Get used to Pytorch
* [x] Implement image classification network from M3 in Pytorch

### Week 2
* [x] Use object detection models in inference
* [x] Train Faster R-CNN on KITTI dataset

### Week 3
* [x] Get familiar with KITTI-MOTS and MOTSChallenge datasets
* [x] Use pre-trained models to evaluate the datasets
* [x] Train Faster R-CNN and RetinaNet on the datasets

### Week 4
* [x] Apply pre-trained Mask-RCNN models to KITTI-MOTS validation set
* [x] Train Mask-RCNN model on KITTI-MOTS training set and evaluate on KITTI-MOTS validation set

### Week 5
* [x] Apply pre-trained and finetuned Mask-RCNN models to MOTSChallenge training set
* [x] Apply pre-trained and finetuned Mask-RCNN models to KITTI-MOTS validation set
* [x] Explore and analyze the impact of different hyperparameters

### Week 6
* [x] Add data augmentation techniques to Detectron2 framework
* [x] Train your model on a synthetic dataset and finetune it on a real dataset
* [x] Train a semantic segmentation model
* [x] Apply tracking techniques for video object segmentation


## Usage 💻
Mini-project:
```
cd mini-project
python3 main.py
```
Week 2:
```
cd week2
python3 train_net.py
```
Week 3:
```
cd week3
python3 train_net.py
```
Week 4:
```
cd week4
python3 predict.py
python3 train_net.py
```
Week 5:
```
cd week4
python3 predict.py
python3 train_net.py
```
Week 6:
```
cd week5
# Data augmentation
python3 train.py

# Tracking
cd tracking
python3 test
```  
Instructions on how to run the deeplab experiments available [here](https://github.com/mariavila/Visual-recognition/blob/master/week6/Semantic_segmentation.md).
  
## Contributors 👫👫
- [Sara Lumbreras Navarro](https://github.com/lunasara) - jfslumbreras@gmail.com
- [Maria Vila](https://github.com/mariavila) - mariava.1213@gmail.com
- [Yael Tudela](https://github.com/yaeltudela) - yaeltudelabarroso@gmail.com
- [Diego Alejandro Velázquez](https://github.com/dvd42) - diegovd0296@gmail.com
