import numpy as np
import tqdm
import argparse
from pathlib import Path
import torch
import cv2
import math
from albumentations.pytorch.functional import img_to_tensor
import logging

# modules
import ds_utils.robseg_2017 as utils
from models.sepconv_model import SepConv

# TODO: add optical flow


def gen_gts_by_interpolate(filenames, interp_type, target_dir, **kwargs):
    # labeled files
    l_filename_first, l_filename_last = filenames[0], filenames[-1]
    l_first = cv2.imread(str(l_filename_first), 0)
    l_last = cv2.imread(str(l_filename_last), 0)
    cv2.imwrite(str(target_dir / l_filename_first.name), l_first)
    cv2.imwrite(str(target_dir / l_filename_last.name), l_last)

    logging.info('first and last: %s, %s' % (str(l_filename_first), str(l_filename_last)))

    # common labels
    common_labels = set.union(set(l_first.flatten()), set(l_last.flatten()))
    common_labels = np.array(sorted(common_labels), dtype=np.int32)

    for idx, ul_filename in enumerate(filenames[1:-1]):
        logging.info('unl: %s' % (str(ul_filename)))
        if interp_type == 'linear':
            # for index:
            # (1-alpha)*a + alpha*b = c
            # e.g. 1 2 3 4 5, for 2: alpha = (2 - 1) / (5 - 1)
            alpha = (idx + 1) / (len(filenames) - 1)

            # generate labels
            unl = (1 - alpha) * l_first + alpha * l_last

            
        elif interp_type == 'sepconv':
            # currently only support 0.5 semi-supervised learning
            assert len(filenames) == 3
            model_sepconv = kwargs['model']
            
            # convert input to correct color space
            tensor_first = img_to_tensor(cv2.cvtColor(l_first, cv2.COLOR_GRAY2BGR))
            tensor_last = img_to_tensor(cv2.cvtColor(l_last, cv2.COLOR_GRAY2BGR))

            tensor_output = estimate_sepconv(model_sepconv, tensor_first, tensor_last)
            # change to BGR
            unl_BGR = (tensor_output.clamp(0.0, 1.0).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            unl = cv2.cvtColor(unl_BGR, cv2.COLOR_BGR2GRAY)

            assert unl.shape == l_first.shape


        else:
            raise NotImplementedError

        # there are several ways to handle invalid labels
        # # 1. filter invalid labels to be background
        # invalid_labels = np.isin(unl, common_labels, invert=True)
        # unl[invalid_labels] = 0


        # 2. quantize based on valid labels
        # e.g.: if valid labels are 0, 1, 3, 7
        # the bins will be [0, 0.5, 1.5, 5]
        # the return of np.digitize will be the indices of bins
        # which helps us to quantize using bins
        bins = []
        for i in range(len(common_labels) - 1):
            # add mean of two labels
            boundary = (common_labels[i] + common_labels[i+1]) / 2.0
            bins.append(boundary)
        quantize_idxs = np.digitize(unl, np.array(bins))
        # quantize according to idxs
        for bin_idx in set(quantize_idxs.flatten()):
            unl[quantize_idxs == bin_idx] = common_labels[bin_idx]


        # write valid gt
        cv2.imwrite(str(target_dir / ul_filename.name), unl.astype(np.uint8))

def estimate_sepconv(model, tensorFirst, tensorSecond):
    assert(tensorFirst.size(1) == tensorSecond.size(1))
    assert(tensorFirst.size(2) == tensorSecond.size(2))

    intWidth = tensorFirst.size(2)
    intHeight = tensorFirst.size(1)

    # assert(intWidth <= 1280) # while our approach works with larger images, we do not recommend it unless you are aware of the implications
    # assert(intHeight <= 720) # while our approach works with larger images, we do not recommend it unless you are aware of the implications

    tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight, intWidth)
    tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(51 / 2.0)) + intWidth + int(math.floor(51 / 2.0))
    intPreprocessedHeight = int(math.floor(51 / 2.0)) + intHeight + int(math.floor(51 / 2.0))

    if intPreprocessedWidth != ((intPreprocessedWidth >> 7) << 7):
        intPreprocessedWidth = (((intPreprocessedWidth >> 7) + 1) << 7) - intPreprocessedWidth # more than necessary
    # end
    
    if intPreprocessedHeight != ((intPreprocessedHeight >> 7) << 7):
        intPreprocessedHeight = (((intPreprocessedHeight >> 7) + 1) << 7) - intPreprocessedHeight # more than necessary
    # end

    tensorPreprocessedFirst = torch.nn.functional.pad(input=tensorPreprocessedFirst, pad=[ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) + intPreprocessedWidth, int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) + intPreprocessedHeight ], mode='replicate')
    tensorPreprocessedSecond = torch.nn.functional.pad(input=tensorPreprocessedSecond, pad=[ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) + intPreprocessedWidth, int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) + intPreprocessedHeight ], mode='replicate')

    return torch.nn.functional.pad(input=model(tensorPreprocessedFirst, tensorPreprocessedSecond), pad=[ 0 - int(math.floor(51 / 2.0)), 0 - int(math.floor(51 / 2.0)) - intPreprocessedWidth, 0 - int(math.floor(51 / 2.0)), 0 - int(math.floor(51 / 2.0)) - intPreprocessedHeight ], mode='replicate')[0, :, :, :].cpu()


def main(args):
    # log level
    logging.basicConfig(level=args.log_level)

    # for sepconv: need to initialize some stuff
    if args.interp_type == 'sepconv':
        torch.set_grad_enabled(False)
        torch.backends.cudnn.enabled = True
        torch.cuda.device(0) # use one GPU
        model_sepconv = SepConv().cuda().eval()
        model_sepconv.load_state_dict(torch.load(args.pretrained_sepconv_model_dir))
        logging.info('load sepconv model complete')

    elif args.interp_type == 'optflow':
        raise NotImplementedError



    train_dir = Path(args.train_dir)
    for ins_id in range(1, 9):
        ins_folder = train_dir / ('instrument_dataset_' + str(ins_id))
        for problem_type, mask_folder in utils.mask_folder.items():
            problem_mask_folder = ins_folder / mask_folder
            # total filenames
            filenames = sorted(list(problem_mask_folder.glob('*.png')))
            num_imgs = len(filenames)
            # labels: 0 (no gt), 1(gt)
            labels = np.zeros((num_imgs,))
            sep = int(1 // args.semi_percentage)
            labels[::sep] = 1
            # the last frame should be 1 too
            labels[-1] = 1
            # index where the file is labeled
            labeled_idxs = np.where(labels == 1)[0]
            for i in range(len(labeled_idxs) - 1):
                labeled_idx1, labeled_idx2 = labeled_idxs[i], labeled_idxs[i+1]
                batch_filenames = filenames[labeled_idx1:labeled_idx2 + 1]
                target_dir = problem_mask_folder.parent / '_'.join([mask_folder, args.interp_type])
                target_dir.mkdir(exist_ok=True, parents=True)
                input_dict = {
                    'filenames': batch_filenames,
                    'interp_type': args.interp_type,
                    'target_dir': target_dir
                }
                if args.interp_type == 'sepconv':
                    input_dict['model'] = model_sepconv
                elif args.interp_type == 'optflow':
                    raise NotImplementedError
                gen_gts_by_interpolate(**input_dict)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='interpolate gts for semi-supervised learning')
    parser.add_argument('--pretrained_optflow_model_dir', type=str, default='../pretrained_model/network-css.pytorch',
        help='directory of UnFlow pretrained model.')
    parser.add_argument('--pretrained_sepconv_model_dir', type=str, default='../pretrained_model/network-lf.pytorch',
        help='directory of sepconv pretrained model.')
    parser.add_argument('--train_dir', type=str, default='../data/cropped_train',
        help='train data directory.')
    parser.add_argument('--interp_type', type=str, default='linear', 
        choices=['linear', 'optflow', 'sepconv'])
    parser.add_argument('--semi_percentage', type=float, default=0.5,
        help='percentage of labeled data for semi-supervised learning.')
    parser.add_argument('--log_level', type=int, default=logging.INFO,
        help='logging level.')
    args = parser.parse_args()
    main(args)    