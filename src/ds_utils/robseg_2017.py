import os
import cv2
import numpy as np
import tqdm
from pathlib import Path
import argparse


"""
utils for MICCAI 2017 robot instrument segmentation dataset
"""

# number of classes for each type of problem
problem_class = {
    'binary': 2,
    'parts': 4,
    'instruments': 8
}

# factor
problem_factor = {
    'binary' : 255,
    'parts' : 85,
    'instruments' : 32
}

# folder names
mask_folder = {
    'binary': 'binary_masks',
    'parts': 'parts_masks',
    'instruments': 'instruments_masks'
}

mask_folder_linear = {
    'binary': 'binary_masks_linear',
    'parts': 'parts_masks_linear',
    'instruments': 'instruments_masks_linear'
}

# self defined train_val split
folds = {0: [1, 3],
         1: [2, 5],
         2: [4, 8],
         3: [6, 7]}

num_frames_video = 225 # number of frames per video
original_height, original_width = 1080, 1920 # original img h, w
cropped_height, cropped_width = 1024, 1280 # cropped img h, w
h_start, w_start = 28, 320 # crop origin



"""
file utilities:
get images, masks, optical flows, predictions, attention maps 
from specific instrument dataset
"""


def get_data(data_dir, data_type, folder_id=-1):
    """
    @params:
    data_type: image, optflows or problem type

    return:
    a list of data of specific type
    """

    global mask_folder
    
    # if it belongs to problem_type, return masks
    if data_type in mask_folder.keys():
        folder_name = mask_folder[data_type]
    else:
        folder_name = data_type

    if folder_id > 0:
        # get dataset<folder_id>
        filenames = (Path(data_dir) / ('instrument_dataset_' + str(folder_id)) \
            / folder_name).glob('*')
    elif folder_id == -1:
        # get all data
        filenames = []
        for folder_id in range(1, 9):
            filenames += (Path(data_dir) / ('instrument_dataset_' + str(folder_id)) \
            / folder_name).glob('*')

    assert len(filenames) > 0
    
    # sort by name
    return list(sorted(filenames))


# TODO
def get_preds(preds_dir, problem_type, folder_id):
    '''
    preds_dir: Pathlib Path object
    '''
    filenames = list(sorted((preds_dir / ('instrument_dataset_' + str(folder_id)) / problem_type).glob('*')))
    return filenames

def get_all_preds(preds_dir, problem_type):
    '''
    preds_dir: Pathlib Path object
    '''
    filenames = []
    for folder_id in range(1, 9):
        filenames += get_preds(preds_dir, problem_type, folder_id)
    return filenames



# set folds[fold] as validation set
# others as training_set
def trainval_split(data_dir, fold):
    global folds

    train_file_names = []
    val_file_names = []

    for idx in range(1, 9):
        # sort according to filename
        filenames = (Path(data_dir) / ('instrument_dataset_' + str(idx)) / 'images').glob('*')
        filenames = list(sorted(filenames))
        if idx in folds[fold]:
            val_file_names += filenames
        else:
            train_file_names += filenames

    return train_file_names, val_file_names



"""
[1] Merge masks with different instruments into one binary mask
[2] Crop black borders from images and masks
"""

def preprocess_data(args):
    # data dir
    data_dir = Path(args.data_dir)
    assert data_dir.exists() == True

    # cropped data dir
    cropped_data_dir = Path(args.cropped_data_dir)
    cropped_data_dir.mkdir(exist_ok=True, parents=True)

    # max number of folders
    num_folders = 8 if args.mode == 'train' else 10

    global problem_factor, mask_folder
    # crop (cropped_height, cropped_width) from (h_start, w_start)
    global cropped_height, cropped_width, h_start, w_start


    for idx in range(1, num_folders + 1):
        # original dataset dir
        instrument_folder = data_dir / ('instrument_dataset_' + str(idx))

        # original mask folder
        ori_mask_folders = list((instrument_folder / 'ground_truth').glob('*'))

        # video frames dir (only read left frames)
        frames_dir = instrument_folder / 'left_frames'

        # processed dataset dir
        processed_instrument_folder = cropped_data_dir / ('instrument_dataset_' + str(idx))

        # mkdir for each problem_type
        image_folder = processed_instrument_folder / 'images'
        image_folder.mkdir(exist_ok=True, parents=True)

        binary_mask_folder = processed_instrument_folder / mask_folder['binary']
        binary_mask_folder.mkdir(exist_ok=True, parents=True)

        parts_mask_folder = processed_instrument_folder / mask_folder['parts']
        parts_mask_folder.mkdir(exist_ok=True, parents=True)

        instrument_mask_folder = processed_instrument_folder / mask_folder['instruments']
        instrument_mask_folder.mkdir(exist_ok=True, parents=True)

        for file_name in tqdm.tqdm(list(frames_dir.glob('*')),
            desc='preprocess dataset %d' % idx, dynamic_ncols=True):
            img = cv2.imread(str(file_name))
            old_h, old_w, _ = img.shape

            img = img[h_start: h_start + cropped_height, w_start: w_start + cropped_width]
            # save cropped frame
            cv2.imwrite(str(image_folder / (file_name.name)), img)
            # create empty masks
            mask_binary = np.zeros((old_h, old_w))
            mask_parts = np.zeros((old_h, old_w))
            mask_instruments = np.zeros((old_h, old_w))

            for ori_mask_folder in ori_mask_folders:
                # read in grayscale
                mask = cv2.imread(str(ori_mask_folder / file_name.name), 0)

                # mark each type of instruments
                # background will be set to 0 in default
                if 'Bipolar_Forceps' in str(ori_mask_folder):
                    mask_instruments[mask > 0] = 1
                elif 'Prograsp_Forceps' in str(ori_mask_folder):
                    mask_instruments[mask > 0] = 2
                elif 'Large_Needle_Driver' in str(ori_mask_folder):
                    mask_instruments[mask > 0] = 3
                elif 'Vessel_Sealer' in str(ori_mask_folder):
                    mask_instruments[mask > 0] = 4
                elif 'Grasping_Retractor' in str(ori_mask_folder):
                    mask_instruments[mask > 0] = 5
                elif 'Monopolar_Curved_Scissors' in str(ori_mask_folder):
                    mask_instruments[mask > 0] = 6
                elif 'Other' in str(ori_mask_folder):
                    mask_instruments[mask > 0] = 7

                # process dir exclude 'Other_labels'
                if 'Other' not in str(ori_mask_folder):
                    # if exists, will be add in
                    mask_binary += mask

                    mask_parts[mask == 10] = 1  # Shaft
                    mask_parts[mask == 20] = 2  # Wrist
                    mask_parts[mask == 30] = 3  # Claspers

            # crop and save masks
            mask_binary = (mask_binary[h_start: h_start + cropped_height, 
                w_start: w_start + cropped_width] > 0).astype(np.uint8) * problem_factor["binary"]
            mask_parts = (mask_parts[h_start: h_start + cropped_height, 
                w_start: w_start + cropped_width]).astype(np.uint8) * problem_factor["parts"]
            mask_instruments = (mask_instruments[h_start: h_start + cropped_height,
                w_start: w_start + cropped_width]).astype(np.uint8) * problem_factor["instruments"]

            cv2.imwrite(str(binary_mask_folder / file_name.name), mask_binary)
            cv2.imwrite(str(parts_mask_folder / file_name.name), mask_parts)
            cv2.imwrite(str(instrument_mask_folder / file_name.name), mask_instruments)
