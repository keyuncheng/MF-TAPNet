import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from pathlib import Path
import tqdm
import argparse
import re
import sys
import logging

from albumentations import (
    Compose,
    Normalize,
    Resize,
    PadIfNeeded,
    VerticalFlip,
    HorizontalFlip,
    RandomCrop,
    CenterCrop,
)

import ignite.engine as engine
import ignite.handlers as handlers
import ignite.contrib.handlers as c_handlers
import ignite.metrics as imetrics


# modules
import ds_utils.robseg_2017 as utils
from loss import LossMulti
from dataset import RobotSegDataset
from metrics import (
    iou_multi_np,
    dice_multi_np,
    class_mean_metric,
    data_mean_metric,
    )
from models.plane_model import *
from models.tap_model import *



def main(args):

    # check cuda available
    assert torch.cuda.is_available() == True


    # model ckpt name prefix
    model_save_dir = '_'.join([args.model, args.problem_type, str(args.lr)])
    # we can add more params for comparison in future experiments
    model_save_dir = '_'.join([model_save_dir, str(args.jaccard_weight), \
         str(args.batch_size), str(args.input_height), str(args.input_width)])

    if args.semi == True:
        model_save_dir = '_'.join([model_save_dir, args.semi_method, str(args.semi_percentage)])

    # model save directory
    model_save_dir = Path(args.model_save_dir) / model_save_dir
    model_save_dir.mkdir(exist_ok=True, parents=True)
    args.model_save_dir = str(model_save_dir)


    # loggers
    logging_logger = logging.getLogger('train')
    # log level
    logging_logger.setLevel(args.log_level)

    # write log to console
    rf_handler = logging.StreamHandler(sys.stderr)
    # rf_handler.setLevel(logging.DEBUG)
    rf_handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s"))
    logging_logger.addHandler(rf_handler)

    # write log to file (fold)
    f_handler = logging.FileHandler(str(model_save_dir /  (args.log_filename + '.log')))
    # f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))
    logging_logger.addHandler(f_handler)
    # add as arguments
    args.logging_logger = logging_logger


    # TODO: add tensorboardX and tf logger in ignite
    # visualization of internal values (attention maps, gated results outputs, etc)
    if args.tb_log:
        from ignite.contrib.handlers import TensorboardLogger
        from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler
        import torchvision.utils as tvutils

        # tensorboard logger
        tf_log_dir = model_save_dir / 'tb_logs'
        tf_log_dir.mkdir(exist_ok=True, parents=True)
        tb_logger = TensorboardLogger(log_dir=str(tf_log_dir))
        # add as arguments
        args.tb_logger = tb_logger



    # input params
    input_msg = 'input arguments: \n'
    for key, val in vars(args).items():
        input_msg += '{}: {}\n'.format(key, val)
    logging_logger.info(input_msg)

    # when the input dimension doesnot change, add this flag to speed up
    cudnn.benchmark = True

    # metrics mean and std
    mean_metrics = {'miou': 0, 'std_miou': 0, 'mdice': 0, 'std_mdice': 0}

    for fold in args.folds:
        metrics_records_fold = train_fold(fold, args)
        # find best validation results by miou
        best_epoch, best_record = sorted(metrics_records_fold.items(), 
            key=lambda item: item[1]['miou'], reverse=True)[0]
        
        logging_logger.info('fold: %d, metrics: %s on epoch %d' % (fold, best_record, best_epoch))
        
        # accumulate metrics and calculate mean for all folds
        for metric_name, val in best_record.items():
            mean_metrics[metric_name] += best_record[metric_name] / len(args.folds)

    logging_logger.info('average on validation for %d folds: \
        mean IoU: %.6e, std: %.6e, mean Dice: %.6e, std: %.6e' % \
        (len(args.folds), mean_metrics['miou'], mean_metrics['std_miou'], \
            mean_metrics['mdice'], mean_metrics['std_mdice']))


def train_fold(fold, args):
    # loggers
    logging_logger = args.logging_logger
    if args.tb_log:
        tb_logger = args.tb_logger

    num_classes = utils.problem_class[args.problem_type]
    # inputs are RGB images (3 * h * w)
    # outputs are 2d multilabel segmentation maps (h * w)
    model = eval(args.model)(in_channels=3, num_classes=num_classes)
    # data parallel for multi-GPU
    model = nn.DataParallel(model, device_ids=args.device_ids).cuda()


    # transform for train/valid data
    train_transform, valid_transform = get_transform(args.model)

    # loss function
    loss_func = LossMulti(num_classes=num_classes, jaccard_weight=args.jaccard_weight)

    # train/valid filenames
    train_filenames, valid_filenames = utils.trainval_split(args.train_dir, fold)


    # additional train dataset kws
    train_ds_kwargs = {
        'filenames': train_filenames,
        'problem_type': args.problem_type,
        'transform': train_transform,
        'model': args.model,
        'mode': 'train',
        'semi': args.semi,
    }


    train_shuffle = True
    valid_num_workers = args.num_workers
    valid_batch_size = args.batch_size
    # additional ds args
    if 'TAPNet' in args.model:
        # for TAPNet, cancel default shuffle, use self-defined shuffle in torch.Dataset instead
        train_shuffle = False
        train_ds_kwargs['batch_size'] = args.batch_size
        # in validation, num_workers should be set to 0 for sequences
        valid_num_workers = 0
        # in validation, batch_size should be set to 1 for sequences
        valid_batch_size = 1

    if args.semi == True:
        train_ds_kwargs['semi_method'] = args.semi_method
        train_ds_kwargs['semi_percentage'] = args.semi_percentage

    # additional valid dataset kws
    valid_ds_kwargs = {
        'filenames': valid_filenames,
        'problem_type': args.problem_type,
        'transform': valid_transform,
        'model': args.model,
        'mode': 'valid',
    }

    # train dataloader
    train_loader = DataLoader(
        dataset=RobotSegDataset(**train_ds_kwargs),
        shuffle=train_shuffle, # set to False to disable pytorch dataset shuffle
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        pin_memory=True
    )
    # valid dataloader
    valid_loader = DataLoader(
        dataset=RobotSegDataset(**valid_ds_kwargs),
        shuffle=False, # in validation, no need to shuffle
        num_workers=valid_num_workers,
        batch_size=valid_batch_size, # in valid time. have to use one image by one
        pin_memory=True
    )

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # # other types of optimizer could be used
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, 
    #     weight_decay=args.weight_decay, nesterov=True)    


    # process function for ignite engine
    def train_step(engine, batch):
        # set model to train
        model.train()
        # clear gradients
        optimizer.zero_grad()
        
        # additional params to feed into model
        add_params = {}

        inputs = batch['input'].cuda(non_blocking=True)
        with torch.no_grad():
            targets = batch['target'].cuda(non_blocking=True)
            # for TAPNet, add attention maps
            if 'TAPNet' in args.model:
                add_params['attmap'] = batch['attmap'].cuda(non_blocking=True)

        outputs = model(inputs, **add_params)

        # TODO: intergrate more choices of semi_percentage in semi-supervised learning
        if args.semi == True:
            # labeled outputs and targets
            labeleds = batch['labeled'] # binary indicators
            l_outputs = outputs[labeled == True]
            l_targets = targets[labeled == True]

            if args.semi_method == 'ignore':
                loss = loss_func(l_outputs, l_targets)
            elif args.semi_method == 'aug_gt':
                # like fully supervised loss
                loss = loss_func(outputs, targets)
            elif args.semi_method == 'rev_flow':
                alpha = args.semi_loss_alpha
                ul_outputs = outputs[labeled == False]
                # ul_targets are targets of previous labeled inputs
                ul_targets = outputs[labeled == False]
                ul_optflows = batch['optflow'][labeled == False]
                # inverse outputs
                inv_outputs = cal_attmaps(ul_outputs, ul_optflows, inverse=True)
                if alpha is None:
                    '''
                    EXPERIMENT: 
                    attention map will be updated for every data
                    but only do backward pass for supervised data
                    '''
                    loss = loss_func(l_outputs, l_targets) 
                else:
                    # linear combination of supervised and unsupervised loss
                    loss = (1 - alpha) * loss_func(l_outputs, l_targets) 
                    + alpha * loss_func(inv_outputs, ul_targets)

        else:
            # fully supervised loss
            loss = loss_func(outputs, targets)

        loss.backward()
        optimizer.step()

        # for TAPNet, update attention maps after each iteration
        if 'TAPNet' in args.model:
            # output_classes and target_classes: <b, h, w>
            output_logsoftmax_np = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            # update attention maps
            train_loader.dataset.update_attmaps(output_logsoftmax_np, batch['idx'].numpy())

        return {
            'loss': loss.item(),
        }
    
    # init trainer
    trainer = engine.Engine(train_step)

    # lr scheduler and handler
    step_scheduler = optim.lr_scheduler.StepLR(optimizer,
        step_size=args.lr_decay_epochs, gamma=args.lr_decay)


    lr_scheduler = c_handlers.param_scheduler.LRScheduler(step_scheduler)
    trainer.add_event_handler(engine.Events.EPOCH_STARTED, lr_scheduler)


    @trainer.on(engine.Events.STARTED)
    def trainer_start_callback(engine):
        logging_logger.info('training fold {}, {} train / {} valid files'. \
            format(fold, len(train_filenames), len(valid_filenames)))

        if args.ckpt_dir is not None:
            # make sure the ckpt_dir exists
            ckpt_dir = Path(args.ckpt_dir)
            assert ckpt_dir.exists() == True

            # ckpt for this fold fold_<fold>_model_<epoch>.pth
            filenames = ckpt_dir.glob('fold_%d_model_[0-9]*.pth' % fold)
            if len(filenames) != 1:
                raise ValueError('invalid model ckpt name. correct ckpt name should be \
                    fold_<fold>_model_<epoch>.pth')

            ckpt_filename = filenames[0]
            res = re.match(r'fold_%d_model_(\d+).pth' % fold, ckpt_filename)
            # restore epoch
            engine.state.epoch = int(res.groups()[0])

            # load state dict
            model.load_state_dict(torch.load(str(ckpt_filename)))
            logging_logger.info('Restored model [{}] from epoch {}.'.format(args.model, engine.state.epoch))
        else:
            logging_logger.info('train model [{}] from scratch'.format(args.model))

        # record metrics history every epoch
        engine.state.metrics_records = {}

        pass

    
    @trainer.on(engine.Events.EPOCH_STARTED)
    def trainer_epoch_start_callback(engine):
        # log learning rate on pbar
        train_pbar.log_message('model: %s, problem type: %s, fold: %d, batch size: %d, lr: %.5e' % \
            (args.model, args.problem_type, fold, args.batch_size, lr_scheduler.get_param()))
        
        # for TAPNet, change dataset schedule to random after the first epoch
        if 'TAPNet' in args.model and engine.state.epoch > 1:
            train_loader.dataset.set_dataset_schedule("shuffle")

        pass


    @trainer.on(engine.Events.ITERATION_COMPLETED)
    def trainer_iter_comp_callback(engine):
        # logging_logger.info(engine.state.output)
        pass

    # monitor loss
    train_ra_loss = imetrics.RunningAverage(output_transform=
        lambda x: x['loss'])
    train_ra_loss.attach(trainer, 'ra_train_loss')

    # metric names
    train_metric_names = ['ra_train_loss']

    train_pbar = c_handlers.ProgressBar(persist=True, dynamic_ncols=True)
    train_pbar.attach(trainer, metric_names=train_metric_names)


    if args.tb_log:
        # attach tensorboard logger
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer, 'lr'), 
            event_name=engine.Events.EPOCH_STARTED)

        tb_logger.attach(trainer, log_handler=OutputHandler('training', train_metric_names),
            event_name=engine.Events.ITERATION_COMPLETED)


    def valid_step(engine, batch):
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

            loss = loss_func(outputs, targets)

            output_logsoftmax_np = torch.softmax(outputs, dim=1).cpu().numpy()
            # output_classes and target_classes: <b, h, w>
            output_classes = output_logsoftmax_np.argmax(axis=1)
            target_classes = targets.data.cpu().numpy()
            
            # record current batch metrics
            b_class_miou = {}
            b_class_mdice = {}
            b_data_miou = []
            b_data_mdice = []

            # loop for each image in batch
            for output_class, target_class in zip(output_classes, target_classes):
                # calculate for each img independently, because class not overlap
                iou_dict = iou_multi_np(target_class, output_class)
                dice_dict = dice_multi_np(target_class, output_class)\

                # sorted by class label
                iou_items = sorted(iou_dict.items(), key=lambda item: item[0])
                dice_items = sorted(dice_dict.items(), key=lambda item: item[0])

                # accumulate batch metrics
                for (cls_iou, iou), (cls_dice, dice) in zip(iou_items, dice_items):
                    assert cls_iou == cls_dice
                    cls = cls_iou
                    # add records for batch
                    b_class_miou[cls] = b_class_miou.get(cls, []) + [iou]
                    b_class_mdice[cls] = b_class_mdice.get(cls, []) + [dice]

                # it's possible that all gt are backgrounds, so neglect those images
                if len(iou_dict) > 0:
                    b_data_miou += [np.mean(list(iou_dict.values()))]
                    b_data_mdice += [np.mean(list(dice_dict.values()))]

            # accumulate batch statistics to state.class_m<metric> and state.data_m<metric>
            for cls in b_class_miou:
                engine.state.class_miou[cls] = engine.state.class_miou.get(cls, []) + b_class_miou[cls]
                engine.state.class_mdice[cls] = engine.state.class_mdice.get(cls, []) + b_class_mdice[cls]
            engine.state.data_miou += b_data_miou
            engine.state.data_mdice += b_data_mdice

            return_dict = {
                'loss': loss.item(),
                'output': outputs,
                'target': targets,
                'class_miou': class_mean_metric(b_class_miou), # for monitoring
                'class_mdice': class_mean_metric(b_class_mdice), # for monitoring
                'data_miou': data_mean_metric(b_data_miou), # for monitoring
                'data_mdice': data_mean_metric(b_data_mdice), # for monitoring
            }

            if 'TAPNet' in args.model:
                # for TAPNet, update attention maps after each iteration
                valid_loader.dataset.update_attmaps(output_logsoftmax_np, batch['idx'].numpy())
                # for TAPNet, return extra internal values
                return_dict['attmap'] = add_params['attmap']

            return return_dict


    # validator engine
    validator = engine.Engine(valid_step)

    # monitor loss
    valid_ra_loss = imetrics.RunningAverage(output_transform=
        lambda x: x['loss'])
    valid_ra_loss.attach(validator, 'ra_valid_loss')

    # monitor data mean metrics
    valid_data_miou = imetrics.RunningAverage(output_transform=
        lambda x: x['data_miou'])
    valid_data_miou.attach(validator, 'd_mIoU')
    valid_data_mdice = imetrics.RunningAverage(output_transform=
        lambda x: x['data_mdice'])
    valid_data_mdice.attach(validator, 'd_mDice')

    # # monitor class mean metrics
    # valid_class_miou = imetrics.RunningAverage(output_transform=
    #     lambda x: x['class_miou'])
    # valid_class_miou.attach(validator, 'c_mIoU')
    # valid_class_mdice = imetrics.RunningAverage(output_transform=
    #     lambda x: x['class_mdice'])
    # valid_class_mdice.attach(validator, 'c_mDice')


    
    # monitoring iou metrics
    cm = imetrics.ConfusionMatrix(num_classes, 
        output_transform=lambda x: (x['output'], x['target']))
    # monitor iou: calculating all examples
    imetrics.IoU(cm, ignore_index=0).attach(validator, 'iou')

    # also record mean IoU for all class (even not exist in gt)
    # this is not counted as a good metric
    # because it averages all examples iou for all class
    # but in some images only a set of classes exists
    mean_iou = imetrics.mIoU(cm, ignore_index=0).attach(validator, 'mean_iou')

    valid_metric_names = ['ra_valid_loss', 'd_mIoU', 'd_mDice']

    valid_pbar = c_handlers.ProgressBar(persist=True, dynamic_ncols=True)
    valid_pbar.attach(validator, metric_names=valid_metric_names)

    # log interal variables(attention maps, outputs, etc.) on validation
    def tb_log_valid_vars(engine, logger, event_name):
        log_tag = 'validation'
        output = engine.state.output
        batch_size = output['output'].shape[0]
        res_grid = tvutils.make_grid(torch.cat([
            output['output'],
            output['target'],
        ]), padding=2, normalize=True, nrows=batch_size).cpu()
        logger.writer.add_image(tag='%s (outputs, targets)' % (log_tag))

        if 'TAPNet' in model_name:
            # log attention maps and other internal values
            inter_vals_grid = tvutils.make_grid(torch.cat([
                output['attmap'],
            ]), padding=2, normalize=True, nrows=batch_size).cpu()
            logger.writer.add_image(tag='%s internal vals' % (log_tag))

    if args.tb_log:
        # log internal values
        tb_logger.attach(validator, log_handler=tb_log_valid_vars, 
            event_name=engine.Events.EPOCH_COMPLETED)
        tb_logger.attach(validator, log_handler=OutputHandler('validation', valid_metric_names),
            event_name=engine.Events.ITERATION_COMPLETED)


    @validator.on(engine.Events.STARTED)
    def validator_start_callback(engine):
        pass

    @validator.on(engine.Events.EPOCH_STARTED)
    def validator_epoch_start_callback(engine):
        '''
        class_m<metric> records <metric> per class per image
        data_m<metric> records mean <metric> per image

        e.g. two images with iou: {1: 0.2, 3: 0.4, 6: 0.2}, {2: 0.3, 3: 0.2 6: 0.1}

        class_m<metric> stores like: 
        {1: [0.2,], 2: [0.3,], 3: [0.4, 0.2,], 6: [0.2, 0.1,]}

        data_m<metric> stores like:
        [np.mean(0.2, 0.4, 0.2), np.mean(0.3, 0.2, 0.1)] = [0.266667, 0.2,]

        when calculating final mean_iou, there are actually two ways:
        refer to function: class_mean_metric and data_mean_metric

        ATTENTION: currently we use data_m<metric>
        '''

        engine.state.class_miou = {}
        engine.state.class_mdice = {}

        engine.state.data_miou = []
        engine.state.data_mdice = []


    # evaluate after epoch finish
    @validator.on(engine.Events.EPOCH_COMPLETED)
    def validator_epoch_comp_callback(engine):
        logging_logger.info(engine.state.metrics)
        ious = engine.state.metrics['iou']
        logging_logger.info('nonzero mean IoU for all data: {:.6e}'.format(ious[ious > 0].mean()))
        # msg = 'IoU: '
        # for ins_id, iou in enumerate(ious):
        #     msg += '{:d}: {:.6f}, '.format(ins_id + 1, iou)
        # logging_logger.info(msg)

        # sort by class label
        class_miou_items = sorted(engine.state.class_miou.items(), key=lambda x: x[0])
        class_mdice_items = sorted(engine.state.class_mdice.items(), key=lambda x: x[0])

        # mean metrics for all valid classes
        msg = 'mean IoU per valid class: '
        for ins_id, ins_ious in class_miou_items:
            msg += '%d: %.6e (%d), ' % (ins_id, np.mean(ins_ious), len(ins_ious))

        msg += 'mean: {:.6e}'.format(class_mean_metric(engine.state.class_miou))
        logging_logger.info(msg)

        msg = 'mean Dice per valid class: '
        for ins_id, ins_dices in class_mdice_items:
            msg += '%d: %.6e (%d), ' % (ins_id, np.mean(ins_dices), len(ins_dices))

        msg += 'mean: {:.6e}'.format(class_mean_metric(engine.state.class_mdice))
        logging_logger.info(msg)

        # mean metrics for all data
        data_miou = data_mean_metric(engine.state.data_miou)
        std_miou = np.std(engine.state.data_miou)
        data_mdice = data_mean_metric(engine.state.data_mdice)
        std_mdice = np.std(engine.state.data_mdice)

        logging_logger.info('data (%d) mean IoU: %.6e, std: %.6e; mean Dice: %.6e, std: %.6e' 
            % (len(engine.state.data_miou), data_miou, std_miou, data_mdice, std_mdice))

        # record metrics in trainer every epoch
        trainer.state.metrics_records[trainer.state.epoch] = \
            {'miou': data_miou, 'std_miou': std_miou, 
            'mdice': data_mdice, 'std_mdice': std_mdice}

        pass

    # evaluate after iter finish
    @validator.on(engine.Events.ITERATION_COMPLETED)
    def validator_iter_comp_callback(engine):
        # batch_iou = engine.state.output['iou']
        # logging_logger.info(np.mean(list(batch_iou.values())))
        # logging_logger.info(np.mean(engine.state.data_miou[-args.batch_size:]))

        pass


    # score function for model saving
    ckpt_score_function = lambda engine: data_mean_metric(engine.state.data_miou)
    ckpt_filename_prefix = 'fold_%d' % fold

    # model saving handler
    model_ckpt_handler = handlers.ModelCheckpoint(
        dirname=args.model_save_dir,
        filename_prefix=ckpt_filename_prefix, 
        score_function=ckpt_score_function,
        create_dir=True,
        require_empty=False,
        save_as_state_dict=True,
        atomic=True)


    validator.add_event_handler(event_name=engine.Events.EPOCH_COMPLETED, 
        handler=model_ckpt_handler,
        to_save={
            'model': model,
        })

    # early stop
    # trainer=trainer, but should be handled by validator
    early_stopping = handlers.EarlyStopping(patience=20, 
        score_function=ckpt_score_function,
        trainer=trainer
        )

    validator.add_event_handler(event_name=engine.Events.EPOCH_COMPLETED,
        handler=early_stopping)


    # evaluate after epoch finish
    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def trainer_epoch_comp_callback(engine):
        validator.run(valid_loader)

    trainer.run(train_loader, max_epochs=args.max_epochs)

    if args.tb_log:
        # close tb_logger
        tb_logger.close()

    return trainer.state.metrics_records



def get_transform(model_name):
    if 'TAPNet' in model_name:
        # transform for sequences of images is very tricky
        # TODO: more transforms should be adopted for better results
        train_transform_ops = [
            PadIfNeeded(min_height=args.input_height, min_width=args.input_width, p=1),
            Normalize(p=1),
            # optional
            Resize(height=args.input_height, width=args.input_width, p=1),
            # CenterCrop(height=args.input_height, width=args.input_width, p=1)
        ]
    else:
        train_transform_ops = [
            PadIfNeeded(min_height=args.input_height, min_width=args.input_width, p=1),
            Normalize(p=1),
            # optional
            # Resize(height=args.input_height, width=args.input_width, p=1),
            # CenterCrop(height=args.input_height, width=args.input_width, p=1)

            # the following transformation should be valid for non-sequence training 
            RandomCrop(height=args.input_height, width=args.input_width, p=1),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5)
        ]

    valid_transform_ops = [
        Normalize(p=1),
        PadIfNeeded(min_height=args.input_height, min_width=args.input_width, p=1),

        # optional
        Resize(height=args.input_height, width=args.input_width, p=1),
        # CenterCrop(height=args.input_height, width=args.input_width, p=1)
    ]
    return Compose(train_transform_ops, p=1), Compose(valid_transform_ops, p=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('--device_ids', type=int, default=[0,1,2,3], nargs='+',
        help='GPU devices ids.')

    parser.add_argument('--num_workers', type=int, default=0,
        help='number of workers for pytorch parallel accleration. 0 for 1 thread.')

    parser.add_argument('--train_dir', type=str, default='../data/cropped_train',
        help='train data directory.')

    parser.add_argument('--batch_size', type=int, default=8,
        help='batch size for input.')

    parser.add_argument('--folds', type=int, default=[0,1,2,3], nargs='+', choices=[0,1,2,3],
        help='folds for training. Muptiple folds are allowed.')

    parser.add_argument('--problem_type', type=str, default='binary', metavar='binary',
         choices=['binary', 'parts', 'instruments'], help='problem types for segmentation.')

    parser.add_argument('--jaccard_weight', type=float, default=0.0, 
        help='jaccard weight [0.0, 1.0] for loss calculation.')

    parser.add_argument('--max_epochs', type=int, default=20,
        help='max epochs for training.')

    parser.add_argument('--lr', type=float, default=1e-5,
        help='learning rate.')

    parser.add_argument('--lr_decay', type=float, default=0.9,
        help='learning rate decay.')

    parser.add_argument('--lr_decay_epochs', type=int, default=5,
        help='number of epochs for every learning rate decay.')

    parser.add_argument('--weight_decay', type=float, default=1e-4,
        help='weight decay.')

    parser.add_argument('--input_height', type=int, default=256,
        help='input image height.')

    parser.add_argument('--input_width', type=int, default=320,
        help='input image width.')

    parser.add_argument('--model', type=str, default='UNet',
        help='model for segmentation.')

    parser.add_argument('--model_save_dir', type=str, default='../model',
        help='model save dir.')

    parser.add_argument('--ckpt_dir', type=str, default=None, 
        help='path to model checkpoint to resume training.')

    parser.add_argument('--semi', type=bool, default=False,
        help='use semi-supervised learning.')

    parser.add_argument('--semi_method', type=str, default=None,
        choices=['ignore', 'aug_gt', 'rev_flow'],
        help='method of semi-supervised learning. Choices are: \
        ignore unlabeled data [ignore], \
        use consecutive labeled gt for internal unlabeled data [aug_gt], \
        use labeled gt and reverse optical flow for unlabeled data [rev_flow]')

    parser.add_argument('--semi_percentage', type=float, default=None,
        help='percentage of labeled data for semi-supervised learning.')

    parser.add_argument('--semi_update', type=bool, default=None,
        help='choose whether to update unsupervised data in backward pass \
         in semi-supervised learning.')

    parser.add_argument('--semi_loss_alpha', type=float, default=None,
        help='the ratio of unsupervised loss. final loss = \
        (1 - alpha) * supervised_loss + alpha * unsupervised_loss')

    parser.add_argument('--tb_log', type=bool, default=False,
        help='use TensorboardLogger to log internal results')

    parser.add_argument('--log_level', type=int, default=logging.INFO,
        help='logging level.')

    parser.add_argument('--log_filename', type=str, default='train',
        help='log file name')

    args = parser.parse_args()
    main(args)
