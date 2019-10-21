import cv2
import numpy as np
import tqdm
from pathlib import Path
import argparse

# modules
# more datasets can be used as seperate modules
import ds_utils.robseg_2015 as utils_2015
import ds_utils.robseg_2017 as utils_2017


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess data following the instructions')
    parser.add_argument('--data_dir', type=str, default='../data/train',
        help='original data directory. This should be organized correctly from the instructions')
    parser.add_argument('--target_data_dir', type=str, default='../data/cropped_train',
        help='output preprocessed data directory')
    parser.add_argument('--dataset', type=str, default='robseg_2017', choices=['robseg_2015', 'robseg_2017'],
        help='name of dataset.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
        help='mode of dataset. (train / test)')
    
    args = parser.parse_args()
    
    if args.dataset == 'robseg_2015':
        utils_2015.preprocess_data(args)
    elif args.dataset == 'robseg_2017':
        utils_2017.preprocess_data(args)
    else:
        raise NotImplementedError()

