import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from pathlib import Path
import tqdm
import argparse
import re
import logging

import ignite.contrib.handlers as c_handlers
import ignite.engine as engine
import ignite.handlers as handlers


# modules
import ds_utils.robseg_2017 as utils
from dataset import RobotSegDataset
from models.plane_model import *
from models.tap_model import *


def main(args):
    # log level
    logging.basicConfig(level=args.log_level)

    # check cuda available
    assert torch.cuda.is_available() == True

    # input params
    input_msg = 'input arguments: \n'
    for key, val in vars(args).items():
        input_msg += '{}: {}\n'.format(key, val)
    logging.info(input_msg)
    # when the input dimension doesnot change, add this flag to speed up
    cudnn.enabled = True
    cudnn.benchmark = True

    for fold in args.folds:
        process_fold(fold, args)


def process_fold(fold, args):
    num_classes = utils.problem_class[args.problem_type]
    factor = utils.problem_factor[args.problem_type]
    # inputs are RGB images (3 * h * w)
    # outputs are 2d multilabel segmentation maps (h * w)
    model = eval(args.model)(in_channels=3, num_classes=num_classes)
    # data parallel for multi-GPU
    model = nn.DataParallel(model, device_ids=args.device_ids).cuda()

    ckpt_dir = Path(args.ckpt_dir)
    # ckpt for this fold fold_<fold>_model_<epoch>.pth
    filenames = ckpt_dir.glob('fold_%d_model_[0-9]*.pth' % fold)
    if len(filenames) != 1:
        raise ValueError('invalid model ckpt name. correct ckpt name should be \
            fold_<fold>_model_<epoch>.pth')

    ckpt_filename = filenames[0]
    # load state dict
    model.load_state_dict(torch.load(str(ckpt_filename)))
    logging.info('Restored model [{}] fold {}.'.format(args.model, fold))

    # segmentation mask save directory
    mask_save_dir = Path(args.mask_save_dir) / ckpt_dir.name
    mask_save_dir.mkdir(exist_ok=True, parents=True)

    eval_transform = Compose([
        Normalize(p=1),
        PadIfNeeded(min_height=args.input_height, min_width=args.input_width, p=1),

        # optional
        Resize(height=args.input_height, width=args.input_width, p=1),
        # CenterCrop(height=args.input_height, width=args.input_width, p=1)
    ], p=1)

    # train/valid filenames,
    # we evaluate and generate masks on validation set
    _, eval_filenames = utils.trainval_split(args.train_dir, fold)


    eval_num_workers = args.num_workers
    eval_batch_size = args.batch_size
    # additional ds args
    if 'TAPNet' in args.model:
        # in eval, num_workers should be set to 0 for sequences
        eval_num_workers = 0
        # in eval, batch_size should be set to 1 for sequences
        eval_batch_size = 1

    # additional eval dataset kws
    eval_ds_kwargs = {
        'filenames': eval_filenames,
        'problem_type': args.problem_type,
        'transform': eval_transform,
        'model': args.model,
        'mode': 'eval',
    }

    # valid dataloader
    eval_loader = DataLoader(
        dataset=RobotSegDataset(**eval_ds_kwargs),
        shuffle=False, # in eval, no need to shuffle
        num_workers=eval_num_workers,
        batch_size=eval_batch_size, # in valid time. have to use one image by one
        pin_memory=True
    )

    # process function for ignite engine
    def eval_step(engine, batch):
        with torch.no_grad():
            model.eval()
            inputs = batch['input'].cuda(non_blocking=True)
            targets = batch['target'].cuda(non_blocking=True)

            # additional arguments
            add_params = {}
            # for TAPNet, add attention maps
            if 'TAPNet' in args.model:
                add_params['attmap'] = batch['attmap'].cuda(non_blocking=True)

            outputs = model(inputs, **add_params)
            output_logsoftmax_np = torch.softmax(outputs, dim=1).cpu().numpy()
            # output_classes and target_classes: <b, h, w>
            output_classes = output_logsoftmax_np.argmax(axis=1)
            masks = (output_classes * factor).astype(np.uint8)

            return_dict = {
                'input_filename': batch['input_filename']
                'mask': masks,
            }

            if 'TAPNet' in args.model:
                # for TAPNet, update attention maps after each iteration
                eval_loader.dataset.update_attmaps(output_logsoftmax_np, batch['idx'].numpy())
                # for TAPNet, return extra internal values
                return_dict['attmap'] = add_params['attmap']

            return return_dict

    # eval engine
    evaluator = engine.Engine(eval_step)

    eval_pbar = c_handlers.ProgressBar(persist=True, dynamic_ncols=True)
    valid_pbar.attach(validator)

    # evaluate after iter finish
    @evaluator.on(engine.Events.ITERATION_COMPLETED)
    def evaluator_epoch_comp_callback(engine):
        # save masks for each batch
        batch_output = engine.state.output
        input_filenames = batch_output['input_filename']
        masks = batch_output['mask']

        for i, input_filename in enumerate(input_filenames):
            mask = cv2.resize(masks[i], dsize=(utils.cropped_width, utils.cropped_height), interpolation=cv2.INTER_AREA)
        
            if pad:
                h_start, w_start = utils.h_start, utils.w_start
                h, w = mask.shape
                # recover to original shape
                full_mask = np.zeros((original_height, original_width))
                full_mask[h_start:h_start + h, w_start:w_start + w] = t_mask
                mask = full_mask

            instrument_folder_name = input_filename.parent.parent.name

            # mask_folder/instrument_dataset_x/problem_type_masks/framexxx.png
            mask_folder = mask_save_dir / instrument_folder_name / utils.mask_folder[args.problem_type]
            mask_folder.mkdir(exist_ok=True, parents=True)
            mask_filename = mask_folder / input_filename.name

            cv2.imwrite(str(mask_filename), mask)

            if 'TAPNet' in args.model:
                attmap = batch_output['attmap'][i]

                attmap_folder = mask_save_dir / instrument_folder_name / '_'.join(args.problem_type, 'attmaps')
                attmap_folder.mkdir(exist_ok=True, parents=True)
                attmap_filename = attmap_folder / input_filename.name

                cv2.imwrite(str(attmap_filename), attmap)


    evaluator.run(eval_loader)







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='segmentation mask generation on validation set')
    parser.add_argument('--device_ids', type=int, default=[0,1,2,3], nargs='+',
        help='GPU devices ids.')
    parser.add_argument('--num_workers', type=int, default=0,
        help='number of workers for pytorch parallel accleration. 0 for 1 thread.')
    parser.add_argument('--batch_size', type=int, default=8,
        help='batch size for input.')
    parser.add_argument('--folds', type=int, default=[0,1,2,3], nargs='+', choices=[0,1,2,3],
        help='folds for evaluation. Muptiple folds are allowed.')
    parser.add_argument('--train_dir', type=str, default='../data/cropped_train',
        help='train data directory.')
    parser.add_argument('--problem_type', type=str, default='binary', metavar='binary',
         choices=['binary', 'parts', 'instruments'], help='problem types for segmentation.')
    parser.add_argument('--input_height', type=int, default=256,
        help='input image height.')
    parser.add_argument('--input_width', type=int, default=320,
        help='input image width.')
    parser.add_argument('--model', type=str, default='UNet',
        help='model for segmentation.')
    parser.add_argument('--ckpt_dir', type=str, required=True, 
        help='path to model checkpoint.')
    parser.add_argument('--mask_save_dir', type=str, default='../valid_masks', 
        help='path to save segmentation masks.')
    parser.add_argument('--pad', type=bool, default=False,
        help='pad the segmentation mask to original ground truth size.')


    args = parser.parse_args()
    main(args)