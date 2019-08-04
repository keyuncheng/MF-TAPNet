MF-TAPNet
=============

[TOC]

----------------------------------------------------

NOTE: This repository is still updating in progress
------

Useful Links
------

[MICCAI 2017 Endoscopic vision challenge](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/)

[Dataset and instructions](
https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/Data/)

[Download dataset and report summary](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/Downloads/)

As required, you need to register an account and get the permission to download the dataset.


Instructions
------

To train and evaluate the TF-MAPNet model, you may follow the instructions.


### Dependencies

```
- Python 3.6
- pytorch 0.4.1+
- pytorch-ignite
- tensorboardX
- albumentations
- opencv-python
- cupy
```

### Folder Structure

The structure of this project will be arranged as follows:

```
(root folder)
├── data
|  ├── train
|  |  ├── instrument_dataset_1
|  |  |  ├── left_frames
|  |  |  ├── right_frames
|  |  |  ├── ......
|  |  ├── instrument_dataset_2
|  |  |  ├── left_frames
|  |  |  ├── right_frames
|  |  |  ├── ......
|  ├── cropped_train
├── src
├── pretrained_model
|  ├── network-css.pytorch
├── ......
```

### Download prerequisites


* Assume the working directory is ``$ROOT_DIR/``. Download the train dataset (2 zips including instrument_dataset 1 to 8). Unzip ``instrument_dataset_X`` into ``data/train`` following the above structure. The train dataset should be arranged as ``$ROOT_DIR/data/train/instrument_dataset_X/...``.

```
$ git clone https://github.com/keyuncheng/MF-TAPNet.git
$ cd MF-TAPNet
$ mkdir data/
$ mkdir data/train
$ unzip instrument_1_4_training.zip -d data/train
$ unzip instrument_5_8_training.zip -d data/train
```

* Download UnFlow pytorch pretrained model for optical flow estimation, then move it to ``$ROOT_DIR/pretrained_model/``.

```
$ wget --timestamping http://content.sniklaus.com/github/pytorch-unflow/network-css.pytorch
$ mkdir pretrained_model
$ mv network-css.pytorch pretrained_model
```

* Switch to source code folder

```
$ cd src/
```

### Preprocess data

```
$ python preprocess_data.py
```

* Choose GPUs

```
$ export CUDA_VISIBLE_DEVICES=a,b,c,d (this could be changed according to training details)
```

* Estimate optical flow for image pairs

We use pretrained UnFlow to estimate optical flow for consecutive image pairs in each surgical video. This step is tricky because the UnFlow model are trained with other datasets (KITTI, 1280 * 384) but we are trying to estimate the optical flow using surgical videos frames in different sizes. In addition, we cannot train-from-scratch/finetune the UnFlow model in a supervised way without dense optical flow as ground truths. For more accurate optical flow estimation, we are trying other methods (unsupervised fine-tuning using surgical videos).

```
$ python gen_optflow.py
```

### Train the model

Arguments for model training in ``train.sh`` are in default settings. You may try other models by modifying it.

```
$ sh train.sh
```

### Still updating