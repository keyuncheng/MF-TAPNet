import torch
from torch import nn
import numpy as np
import tqdm

import logging

def iou_binary_torch(y_true, y_pred):
    """
    y_true: 4d tensor <b,c=1,h,w>
    y_pred: 4d tensor <b,c=1,h,w>

    output: 1d tensor <b,c=1>
    """
    assert y_true.dim() == 4 or y_true.dim() == 3
    assert y_pred.dim() == 4 or y_true.dim() == 3

    epsilon = 1e-15
    # sum for dim (h, w)
    intersection = (y_pred * y_true).sum(dim=[-2, -1])
    union = y_true.sum(dim=[-2, -1]) + y_pred.sum(dim=[-2, -1])

    return (intersection + epsilon) / (union - intersection + epsilon)

def dice_binary_torch(y_true, y_pred):
    """
    y_true: 4d tensor <b,c=1,h,w>
    y_pred: 4d tensor <b,c=1,h,w>

    output: 1d tensor <b,c=1>
    """
    assert y_true.dim() == 4 or y_true.dim() == 3
    assert y_pred.dim() == 4 or y_true.dim() == 3

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


# mean metric functions
def class_mean_metric(class_mmetric_dict):
    """
    first calculate mean per class for all data,
    then calculate mean for all classes

    input: dict of <class, list<mean_metric>>
    e.g. {
            1: [mean_iou_cls1(d0), mean_iou_cls1(d2)],
            2: [mean_iou_cls2(d0)],
            4: [mean_iou_cls4(d0), mean_iou_cls4(d1), mean_iou_cls4(d2)],
        }

    output(float): class_mean(data_mean)
    """
    assert isinstance(class_mmetric_dict, dict)
    return np.mean([np.mean(ins_metrics) \
        for ins_metrics in class_mmetric_dict.values()])

def data_mean_metric(data_mmetric_list):
    '''
    first calculate mean per data for all valid classes
    then calculate mean for all data

    input: list of mean_metric
    e.g. [mean_iou(d0), mean_iou(d1), mean_iou(d2), ...]

    output(float): data_mean
    '''
    assert isinstance(data_mmetric_list, list)
    return np.mean(data_mmetric_list)


####################################################################
# not used
def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
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


def get_iou_multi(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious


def get_dice_multi(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 1
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices
