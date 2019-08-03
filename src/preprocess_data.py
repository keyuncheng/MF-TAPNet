import cv2
import numpy as np
import tqdm
from pathlib import Path
import argparse

# modules
# more datasets can be used as seperate modules
import ds_utils.robseg_2017 as utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess data following the instructions')
    parser.add_argument('--data_dir', type=str, default='../data/train',
        help='original data directory. This should be organized correctly from the instructions')
    parser.add_argument('--cropped_data_dir', type=str, default='../data/cropped_train',
        help='output preprocessed data directory')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
        help='mode of dataset. (train / test)')
    args = parser.parse_args()
    
    utils.preprocess_data(args)
