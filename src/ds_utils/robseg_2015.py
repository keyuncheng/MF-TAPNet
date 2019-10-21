#coding=utf-8
import numpy as np
import cv2
# import torch
import tqdm
from pathlib import Path

"""
utils for MICCAI 2015 robot instrument segmentation dataset
"""

num_frames_video = 55 # number of frames per video
height, width = 576, 720

num_videos = {
    'train': 4,
    'test': 6,
}

# number of classes for each type of problem
problem_class = {
    'binary': 2,
    'parts': 4,
    'instruments': None # invalid
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

# for linear interpolation
mask_folder_linear = {
    'binary': 'binary_masks_linear',
    'parts': 'parts_masks_linear',
    'instruments': 'instruments_masks_linear'
}


"""
file utilities:
get images, masks, optical flows, predictions, attention maps 
from specific instrument dataset
"""

def get_data(data_dir, data_type, mode='train', folder_id=-1):
    """
    @params:
    data_type: image, optflows or problem type

    return:
    a list of data of specific type
    """

    global num_videos, mask_folder

    num_folders = num_videos[mode]
    
    # if it belongs to problem_type, return masks
    if data_type in mask_folder.keys():
        folder_name = mask_folder[data_type]
    else:
        folder_name = data_type

    filenames = []

    if folder_id > 0:
        # get instrument_dataset_<folder_id>
        filenames += (Path(data_dir) / ('instrument_dataset_' + str(folder_id)) \
            / folder_name).glob('*')
    elif folder_id == -1:
        # get all data
        for folder_id in range(1, num_folders + 1):
            filenames += (Path(data_dir) / ('instrument_dataset_' + str(folder_id)) \
            / folder_name).glob('*')

    if len(filenames) <= 0:
        raise ValueError("Empty folder, data_type: %s, mode: %s, folder_id: %d"
            % (data_type, mode, folder_id))
    
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



def preprocess_dataset(folder_idx, ds_folder, ds_gt_folder, target_ds_folder, num_frames):

    global num_frames_video
    interval = num_frames / num_frames_video

    video_fn = ds_folder / 'Video.avi'
    assert video_fn.exists() == True
    
    target_img_folder = target_ds_folder / 'images'
    target_img_folder.mkdir(exist_ok=True, parents=True)
    target_gt_binary_folder = target_ds_folder / 'binary'
    target_gt_binary_folder.mkdir(exist_ok=True, parents=True)
    target_gt_parts_folder = target_ds_folder / 'parts'
    target_gt_parts_folder.mkdir(exist_ok=True, parents=True)

    # # video
    vc = cv2.VideoCapture(str(video_fn))
    for i in tqdm.trange(num_frames, desc='preprocess video %d' % folder_idx, dynamic_ncols=True):
        ret,frame = vc.read()
        if i % interval != 0:
            continue
        cv2.imwrite(str(target_img_folder / ('frame%d.png' % i)), frame)
    vc.release()

    # gt
    if (ds_gt_folder / 'Segmentation.avi').exists():
        # only one gt video exists
        video_gt_fn = ds_gt_folder / 'Segmentation.avi'
        vc = cv2.VideoCapture(str(video_gt_fn))
        for i in tqdm.trange(num_frames, desc='preprocess gt %d' % folder_idx, dynamic_ncols=True):
            ret,frame = vc.read()
            if i % interval != 0:
                continue
            # binary gt
            binary_gt = np.zeros_like(frame)
            binary_gt[frame > np.array([0,0,0])] = 255
            parts_gt = np.zeros_like(frame)
            parts_gt[frame == np.array([160,160,160])] = 85 # Shaft
            # manipulator, classified as Wrist (no way to handle this since can be classified as Wrist and Clasper)
            parts_gt[frame == np.array([70,70,70])] = 170

            cv2.imwrite(str(target_gt_binary_folder / ('frame%d.png' % i)), binary_gt)
            cv2.imwrite(str(target_gt_parts_folder / ('frame%d.png' % i)), parts_gt)
        vc.release()
    else:
        # left and right
        video_gt_left_fn = ds_gt_folder / 'Left_Instrument_Segmentation.avi'
        video_gt_right_fn = ds_gt_folder / 'Right_Instrument_Segmentation.avi'

        vc_left = cv2.VideoCapture(str(video_gt_left_fn))
        vc_right = cv2.VideoCapture(str(video_gt_right_fn))
        for i in tqdm.trange(num_frames, desc='preprocess gt l/r', dynamic_ncols=True):
            ret_left ,frame_left = vc_left.read()
            ret_right ,frame_right = vc_right.read()
            if i % interval != 0:
                continue
            binary_gt = np.zeros_like(frame_left)
            binary_gt[frame_left > np.array([0,0,0])] = 255
            binary_gt[frame_right > np.array([0,0,0])] = 255
            parts_gt = np.zeros_like(frame_left)
            parts_gt[frame_left == np.array([160,160,160])] = 85 # Shaft
            parts_gt[frame_right == np.array([160,160,160])] = 85 # Shaft
            # manipulator, classified as Wrist (no way to handle this since can be classified as Wrist and Clasper)
            parts_gt[frame_left == np.array([70,70,70])] = 170
            parts_gt[frame_right == np.array([70,70,70])] = 170

            cv2.imwrite(str(target_gt_binary_folder / ('frame%d.png' % i)), binary_gt)
            cv2.imwrite(str(target_gt_parts_folder / ('frame%d.png' % i)), parts_gt)

        vc_left.release()
        vc_right.release()


'''
[1] capture frames from videos
[2] merge left/right gts to grayscale masks
'''
def preprocess_data(args):
    '''
    cutting videos to frames
    '''
    data_dir = Path(args.data_dir)
    assert data_dir.exists() == True
    target_data_dir = Path(args.target_data_dir)
    if args.mode == 'train':
        root_folder = data_dir / 'Segmentation_Robotic_Training' / 'Training'
        root_gt_folder = data_dir / 'Segmentation_Robotic_Training' / 'Training'
        for i in range(1, 5):
            # train dir
            ds_folder = root_folder / ('Dataset' + str(i))
            ds_gt_folder = root_gt_folder / ('Dataset' + str(i))
            # target dir
            target_ds_folder = target_data_dir / ('instrument_dataset_' + str(i))
            target_ds_folder.mkdir(exist_ok=True, parents=True)
            
            preprocess_dataset(i, ds_folder, ds_gt_folder, target_ds_folder, 1100) # 1100 frames

    elif mode == 'test':
        root_folder = data_dir / 'Segmentation'
        root_gt_folder = data_dir / 'GT' # need to manually create a dir called GT
        for i in range(1, 7):
            # test dir
            ds_folder = root_folder / ('Dataset' + str(i))
            ds_gt_folder = root_gt_folder / ('Dataset' + str(i))
            # target dir
            target_ds_folder = target_data_dir / ('instrument_dataset_' + str(i))
            target_ds_folder.mkdir(exist_ok=True, parents=True)
            num_frames = 370 if i < 5 else 1500
            preprocess_dataset(i, ds_folder, ds_gt_folder, target_ds_folder, num_frames)
    else:
        raise ValueError('invalid mode')
