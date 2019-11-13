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
from visualize import flow_vis, flow_vis_arr


'''
parts of this implementation are borrowed from pytorch UnFlow:
ref: https://github.com/sniklaus/pytorch-unflow
'''

def estimate(model, first, second):
    '''
    given image pair <first, second> (h, w), estimate the dense optical flow (h, w, 2)

    @param first: first images batch 4d-tensor (b, c=3, h, w)
    @param second: second images batch 4d-tensor (b, c=3, h, w)
    return: optical flow for pairs batch 4d-tensor (b, c=2, h, w)
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


class ImagePairsDataset(Dataset):
    """
    ImagePairsDataset: return image pairs
    """

    def __init__(self, filenames):
        super(ImagePairsDataset, self).__init__()
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # an exception for each video:
        # for the first frame in each video, the optical flow should be 0
        # optflow[0] = flow<0, 0> = 0, optflow[k] = flow<k-1, k> (k > 0)
        first_idx = idx if idx % utils.num_frames_video == 0 else idx - 1
        next_idx = idx
        file1, file2 = self.filenames[first_idx], self.filenames[next_idx]
        # according to the UnFlow implementation, inputs are in normalized BGR space
        first = cv2.imread(str(file1))
        second = cv2.imread(str(file2))
        # img_to_tensor will reshape into (c, h, w) and scaled to [0., 1.]
        return str(file2), img_to_tensor(first), img_to_tensor(second)


def main(args):
    torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
    torch.cuda.device(0) # use one GPU

    # load model
    model = UnFlow().cuda().eval()
    model_path = args.pretrained_model_dir
    model.load_state_dict(torch.load(model_path))

    # images
    filenames = utils.get_data(data_dir=args.train_dir, data_type="images")
    # filenames = list(Path(args.train_dir).glob('*'))
    batch_size = args.batch_size
    # dataloader
    loader = DataLoader(
            dataset=ImagePairsDataset(filenames),
            shuffle=False, # no need to shuffle
            num_workers=0, # pretrained model not support parallel
            batch_size=batch_size,
            pin_memory=True
        )
    
    # progress bar
    tq = tqdm.tqdm(total=len(loader.dataset),
        desc='estimate optical flow for image pairs')
    
    for i, (filenames, firsts, seconds) in enumerate(loader):
        outputs = estimate(model, firsts, seconds)

        for filename, output in zip(filenames, outputs):
            flow_uv = output.numpy().transpose(1, 2, 0) # (h, w, c)
            filename = Path(filename)

            # save optical flow in instruments_dataset_X/optflows/filename.flo
            video_dir = filename.parent.parent
            optflow_dir = video_dir / args.optflow_dir
            optflow_dir.mkdir(exist_ok=True, parents=True)
            optfilename = optflow_dir / (filename.stem + ".flo")
            objectOutput = open(str(optfilename), 'wb')
            np.array([80, 73, 69, 72], np.uint8).tofile(objectOutput)
            np.array([output.size(2), output.size(1)], np.int32).tofile(objectOutput)
            # store in (h, w, c)
            np.array(flow_uv, np.float32).tofile(objectOutput)
            objectOutput.close()

            if (args.visualize):
                # save optical flow visualization in color model
                # in instruments_dataset_X/optflows/filename
                optflow_vis_color_dir = video_dir / args.optflow_vis_color_dir
                optflow_vis_color_dir.mkdir(exist_ok=True, parents=True)
                flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
                cv2.imwrite(str(optflow_vis_color_dir / (filename.name)), flow_color)

                # save optical flow visualization in arrows
                # in instruments_dataset_X/optflows/filename
                optflow_vis_arrow_dir = video_dir / args.optflow_vis_arrow_dir
                optflow_vis_arrow_dir.mkdir(exist_ok=True, parents=True)
                flow_arrow = flow_vis_arr.flow_to_arrow(flow_uv)
                cv2.imwrite(str(optflow_vis_arrow_dir / (filename.name)), flow_arrow)
                
        tq.update(batch_size)

    tq.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='use UnFlow to calculate optical flow for train data')
    parser.add_argument('--pretrained_model_dir', type=str, default='../pretrained_model/network-css.pytorch',
        help='directory of UnFlow pretrained model.')
    parser.add_argument('--batch_size', type=int, default=8,
        help='batch size for calculating optical flow.')
    parser.add_argument('--visualize', type=bool, default=True,
        help='store the visualization of the optical flow')
    # dirname
    parser.add_argument('--train_dir', type=str, default='../data/cropped_train',
        help='train data directory.')
    parser.add_argument('--optflow_dir', type=str, default='optflows',
        help='optical flow file save dir. e.g. .../instrument_dataset_X/optflow_dir/*.flo')
    parser.add_argument('--optflow_vis_color_dir', type=str, default='optflows_vis_color',
        help='visualization of optical flow in color model save dir. e.g. .../instrument_dataset_X/optflow_vis_color_dir/*.png')
    parser.add_argument('--optflow_vis_arrow_dir', type=str, default='optflows_vis_arrow',
        help='visualization of optical flow in arrows save dir. e.g. .../instrument_dataset_X/optflow_vis_arrow_dir/*.png')
    args = parser.parse_args()
    main(args)
