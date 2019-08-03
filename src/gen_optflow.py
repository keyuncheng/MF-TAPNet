#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.functional import img_to_tensor
import math
import numpy as np
import sys
import cv2
from pathlib import Path
import tqdm
import argparse


# modules
from models.unflow_model import UnFlow
import ds_utils.robseg_2017 as utils


'''
parts of this implementation are borrowed from pytorch UnFlow:
ref: https://github.com/sniklaus/pytorch-unflow
'''

def estimate(model, first, second):
    '''
    first, second: 4d-tensor (b, c=3, h, w)
    return: optical flow 4d-tensor (b, c=2, h, w)
    '''

    assert first.shape == second.shape

    h, w = first.size()[2:]

    # the default input for UnFlow pretrained model is (h * w) = (384 * 1280)
    # for custom input size, you can resize the input to (384 * 1280) and then resize back
    # just input the original shape is OK, but not guaranteed for correctness
    # comment the following lines for unresize
    default_h = 384
    default_w = 1280
    first = interpolate(input=first, size=(default_h, default_w), mode='bilinear', align_corners=True).cuda()
    second = interpolate(input=second, size=(default_h, default_w), mode='bilinear', align_corners=True).cuda()

    assert first.shape[2] == 384
    assert first.shape[3] == 1280

    output = model(first.cuda(), second.cuda())

    # resize back to input shape
    output = interpolate(input=output, size=(h, w), mode='bilinear', align_corners=True)

    return output.cpu()


class RobotSegDataset(Dataset):
    """docstring for RobotSegDataset"""
    def __init__(self, filenames):
        super(RobotSegDataset, self).__init__()
        self.filenames = filenames

    def __len__(self):
        # num of imgs
        return len(self.filenames)

    def __getitem__(self, idx):
        # an exception for each video:
        # for the first frame in each video, the optical flow should be 0
        # optflow[0] = flow<0, 0> = 0
        # optflow[1] = flow<0, 1>
        first_idx = idx if idx % utils.num_frames_video == 0 else idx - 1
        next_idx = idx
        file1, file2 = self.filenames[first_idx], self.filenames[next_idx]
        # first = cv2.cvtColor(cv2.imread(str(file1)), cv2.COLOR_BGR2RGB)
        # second = cv2.cvtColor(cv2.imread(str(file2)), cv2.COLOR_BGR2RGB)
        # according to the implementation, don't need to change color space since already in BGR
        first = cv2.imread(str(file1))
        second = cv2.imread(str(file2))
        # img_to_tensor will help us move channel axis and div by 255
        return str(file2), img_to_tensor(first), img_to_tensor(second)


def main(args):
    # first reverse the input last dimension?
    # change dimension order
    # then normalize

    torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
    torch.cuda.device(0) # use one GPU

    # load model
    model = UnFlow().cuda().eval()
    model_path = args.pretrained_model_dir
    model.load_state_dict(torch.load(model_path))

    filenames = utils.get_data(data_dir=args.train_dir, data_type="images")

    batch_size = args.batch_size

    loader = DataLoader(
            dataset=RobotSegDataset(filenames),
            shuffle=False,
            num_workers=0,
            batch_size=batch_size,
            pin_memory=True
        )
    
    # progress bar
    tq = tqdm.tqdm(total=len(loader.dataset),
        desc='calculate optical flow')
    
    for i, (filenames, firsts, seconds) in enumerate(loader):
        outputs = estimate(model, firsts, seconds)

        for filename, output in zip(filenames, outputs):
            # save as .flo format in the same parent_dir as original image
            # instruments_dataset_X/images/*.png
            # instruments_dataset_X/optflows/*.flo
            optfilename = Path(filename.replace('images', 'optflows').replace('png', 'flo'))
            optfilename.parent.mkdir(exist_ok=True, parents=True)

            objectOutput = open(str(optfilename), 'wb')

            # save as .flo format
            np.array([80, 73, 69, 72], np.uint8).tofile(objectOutput)
            np.array([output.size(2), output.size(1)], np.int32).tofile(objectOutput)
            np.array(output.numpy().transpose(1, 2, 0), np.float32).tofile(objectOutput)

            objectOutput.close()

        tq.update(batch_size)
    tq.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='use UnFlow to calculate optical flow for train data')
    parser.add_argument('--pretrained_model_dir', type=str, default='../pretrained_model/network-css.pytorch',
        help='directory of UnFlow pretrained model.')
    parser.add_argument('--train_dir', type=str, default='../data/cropped_train',
        help='train data directory.')
    parser.add_argument('--batch_size', type=int, default=8,
        help='batch size for calculating optical flow.')
    args = parser.parse_args()
    main(args)
