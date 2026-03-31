# NYCU Computer Vision 2026 HW1
- Student ID: 314553046
- Name: 朱心慈

## Introduction
This project implements a deep learning model for a 100-class image classification task on a pet image dataset. The goal is to classify each image into its corresponding category using a ResNet-based architecture.

The model is built on a pretrained ResNet-101 backbone and is trained with common data augmentation techniques to improve generalization. In addition, CutMix is applied during training to further enhance robustness by encouraging the model to learn from multiple regions within images.

The implementation is based on PyTorch and includes a complete pipeline for training, validation, and inference, with outputs formatted for competition submission.
## Environment Setup
Install all dependencies.
```
pip install -r requirements.txt
```
## Usage
### Training
```
python main.py --mode train --data_root ./data --cutmix
```
### Inference
```
python main.py --mode inference --data_root ./data --ckpt best_model_cutmix.pth
```
## Performance Snapshot
<img width="1145" height="57" alt="image" src="https://github.com/user-attachments/assets/7ea5f186-ab92-48e1-8208-326bb8026495" />
<img width="379" height="486" alt="image" src="https://github.com/user-attachments/assets/99651d8f-a682-48a4-bfce-7862ed4fb0eb" />
