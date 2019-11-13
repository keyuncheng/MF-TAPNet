import torch
from torch import nn
import numpy as np
import tqdm
import logging


def iou_binary_torch(y_true, y_pred):
    """
    @param y_true: 3d tensor <b,h,w>
    @param y_pred: 3d tensor <b,h,w>

    @return output: 1d tensor <b,c=1>
    """
    assert y_true.shape == y_pred.shape
    assert y_true.dim() == 3

    epsilon = 1e-15
    # sum for dim (h, w)
    intersection = (y_pred * y_true).sum(dim=[-2, -1])
    union = y_true.sum(dim=[-2, -1]) + y_pred.sum(dim=[-2, -1])

    return (intersection + epsilon) / (union - intersection + epsilon)

def dice_binary_torch(y_true, y_pred):
    """
    @param y_true: d tensor <b,h,w>
    @param y_pred: d tensor <b,h,w>

    @return output: 1d tensor <b,c=1>
    """
    assert y_true.shape == y_pred.shape
    assert y_true.dim() == 3

    epsilon = 1e-15
    # sum for dim (h, w)
    intersection = (y_pred * y_true).sum(dim=[-2, -1])
    union = y_true.sum(dim=[-2, -1]) + y_pred.sum(dim=[-2, -1])

    return (2 * intersection + epsilon) / (union + epsilon)


def iou_multi_np(y_true, y_pred):
    '''
    y_true: 2d ndarray <h,w>
    y_pred: 2d ndarray <h,w>

    output: dict {class_id: class_iou}
    '''

    assert y_true.ndim == 2
    assert y_pred.ndim == 2

    result = {}

    # only calculate all labels preseneted in gt, ignore background
    for instrument_id in set(y_true.flatten()):
        result[instrument_id] = iou_binary_np(y_true == instrument_id, y_pred == instrument_id)

    # background with index 0 should not be counted
    result.pop(0, None)

    return result


# for numpy
def dice_multi_np(y_true, y_pred):
    '''
    y_true: 2d ndarray <h,w>
    y_pred: 2d ndarray <h,w>

    output: dict {class_id: class_iou}
    '''

    assert y_true.ndim == 2
    assert y_pred.ndim == 2

    result = {}

    # only calculate all labels preseneted in gt, ignore background
    for instrument_id in set(y_true.flatten()):
        result[instrument_id] = dice_binary_np(y_true == instrument_id, y_pred == instrument_id)

    # background with index 0 should not be counted
    result.pop(0, None)

    return result


def iou_binary_np(y_true, y_pred):
    """
    y_true: 2d tensor <h,w>
    y_pred: 2d tensor <h,w>

    output: float
    """
    assert y_true.ndim == 2
    assert y_pred.ndim == 2


    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice_binary_np(y_true, y_pred):
    """
    y_true: 2d tensor <h,w>
    y_pred: 2d tensor <h,w>

    output: float
    """

    assert y_true.ndim == 2
    assert y_pred.ndim == 2

    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)
        

'''
use confusion matrix to calculate iou and dice
'''
def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    '''
    @param prediction: 3d ndarray classification results (b, h, w)
    @param ground_truth: 3d ndarray gts (b, h, w)
    @param nr_labels: number of classes

    @return confusion matrix
    '''
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def calculate_iou(confusion_matrix):
    ious = {}
    # ignore cls 0 (background)
    for cls in range(1, confusion_matrix.shape[0]):
        true_positives = confusion_matrix[cls, cls]
        false_positives = confusion_matrix[:, cls].sum() - true_positives
        false_negatives = confusion_matrix[cls, :].sum() - true_positives

        # discard the class which are not presented in gt
        if (true_positives + false_negatives) == 0:
            continue

        ious[cls] = float(true_positives) / (true_positives + false_positives + false_negatives)
    return ious


def calculate_dice(confusion_matrix):
    dices = {}
    # ignore cls 0 (background)
    for cls in range(1, confusion_matrix.shape[0]):
        true_positives = confusion_matrix[cls, cls]
        false_positives = confusion_matrix[:, cls].sum() - true_positives
        false_negatives = confusion_matrix[cls, :].sum() - true_positives

        # discard the class which are not presented in gt
        if (true_positives + false_negatives) == 0:
            continue

        dices[cls] = 2 * float(true_positives) / (2 * true_positives + false_positives + false_negatives)
    return dices


class MetricRecord():
    '''
    docs describing the format of _records
    '''
    def __init__(self):
        super(MetricRecord, self).__init__()
        self._records = []

    def update_record(self, record):
        # Note that some record may be empty (no valid (non-background) labels), ignore those samples
        self._records.append(record)

    def merge(self, mRecord):
        if not isinstance(mRecord, MetricRecord):
            raise TypeError('invalid type %s' % type(mRecord))
        self._records += mRecord._records


    def data_mean(self):
        '''
        We adopted this evaluation criteria for each metric
        
        please refer to the challenge summary
        Ref: https://arxiv.org/abs/1902.06426
        Section IV: RESULTS, A. Evaluation Criteria

        1. calculate **arithmatic mean** for all classes that are presented in a frame,
           if we are considering a set of classes and none are present in a frame,
           we **discount** the frame from the calculation
        2. compute this score for each frame and 
            **average** over all frames to get a per-dataset score
        '''

        # calculate mean metric for each record
        data_metrics = [np.mean(list(record.values())) for record in self._records if len(record) > 0]

        if len(data_metrics) == 0:
            data_mean = 0
            data_std = 0
        else:
            data_mean = np.mean(data_metrics)
            data_std = np.std(data_metrics)
        
        # average over all records
        return {
            'items': data_metrics,
            'mean': data_mean,
            'std': data_std
        }


    # # mean metric functions
    # def class_mean(self):
    #     '''
    #     for each class, mean over all samples, then mean over all classes
    #     '''
    #     class_metrics = {}
    #     for record in self._records:
    #         for ins_id, metric in record.items():
    #             class_metrics[ins_id] = class_metrics.get(ins_id, []) + [metric] # append the metric
    #     class_metrics_items = sorted(class_metrics.items(), key=lambda item: item[0])
    #     class_mmetrics_items = [np.mean(metrics) for ins_id, metrics in class_metrics_items]
    #     return {
    #         'items': class_metrics_items,
    #         'mean': np.mean(class_mmetrics_items),
    #         'std': np.std(class_mmetrics_items)
    #     }

