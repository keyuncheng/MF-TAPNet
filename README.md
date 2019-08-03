=============
NOTE: Update in progress


MICCAI 2017 Endoscopic vision challenge
https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/

Dataset and instructions
https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/Data/

Download dataset and summary report:
https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/Downloads/
As required, you need to register an account and get the permission to download the dataset.


Requirements:
- Python 3.6
- pytorch 0.4.1+
- pytorch-ignite
- tensorboardX
- albumentations
- opencv-python
- cupy


The structure of this project will be arranged as follows:

(root folder)
|-data
|  |-train
|  |  |-instrument_dataset_1
|  |  |  |-left_frames
|  |  |  |-right_frames
|  |  |  |-......
|  |  |-instrument_dataset_2
|  |  |  |-left_frames
|  |  |  |-right_frames
|  |  |  |-......
|  |-cropped_train
|-src
|-pretrained_model
|  |-network-css.pytorch
|-......



Instructions

The follow instructions are valid under Ubuntu

Download project and dataset

- we assume the current directory as $ROOT_DIR/
- download the training dataset (2 zips including instrument_dataset_1-8)
- unzip instrument_dataset_X into data/train (X=1-8) following the above structure
- the training dataset should be arranged as $ROOT_DIR/data/train/instrument_dataset_X/...

- $ git clone https://github.com/keyuncheng/MF-TAPNet.git
- $ cd MF-TAPNet
- $ mkdir data/
- $ mkdir data/train
- $ unzip instrument_1_4_training.zip -d data/train
- $ unzip instrument_5_8_training.zip -d data/train


Download pretrained UnFlow pytorch model

- download UnFlow pytorch pretrained model and move it to $ROOT_DIR/pretrained_model/


- $ wget --timestamping http://content.sniklaus.com/github/pytorch-unflow/network-css.pytorch
- $ mkdir pretrained_model
- $ mv network-css.pytorch pretrained_model


switch to source code folder
- $ cd src/


Preprocess data

- $ python preprocess_data.py


Choose GPUs

export CUDA_VISIBLE_DEVICES=a,b,c,d (this could be changed according to training details)


calculate optical flow for each two image pairs using UnFlow

- this step is tricky because the original size of input should not be self-defined,
instead should be like the one it was pre-trained with other datasets (KITTI, 1280 * 384)
- since we cannot train it directly given no optical flow ground truths of our dataset
- for better results of calculating optical flow, we may consider other ways

- $ python gen_optflow.py

model training

- arguments are defined in train.sh and can be modified

- $ sh train.sh


----------------------------------------------------------------------------------------
Problems:
1. how to deal with attention maps when training paused and restored?
2. Sigmoid and Softmax in Attention Module (Not trained)
3. bn in train and valid
4. (IMPORTANT) whether using softmax operation in attention module or not
5. (IMPORTANT)use + or * or conv2d(kernel 1) in final attention module operation

Handle:
1. Dice in binary classification
2. metrics for multiclass classification (Not Debuged)
3. larger Attention map
4. multiple config.py (for preprocess_data.py)
5. Semi-supervised model


TRAIN_RECORD for each model:
Binary: 
1. UNet (not resized) batchsize=8, learn_rate=1e-5, epoches=20, gpu=4 (TITAN V), validation IoU=0.6728385997379875: validation Dice=0.784111333214352
2. UNet11 (not resized) batchsize=8, learn_rate=1e-5, epoches=20, gpu=4 (TITAN V), validation IoU=0.8175544938356124: validation Dice=0.8885956894127561
2. UNet16 (not resized) batchsize=8, learn_rate=1e-5, epoches=20, gpu=4 (TITAN V), validation IoU=: validation Dice=