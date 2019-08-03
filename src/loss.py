import torch
from torch import nn
import numpy as np
import tqdm

from metrics import iou_binary_torch, dice_binary_torch, \
                    iou_multi_np, dice_multi_np, \
    calculate_confusion_matrix_from_arrays, get_iou_multi, get_dice_multi


# parts of implementation are borrowed from:
# https://github.com/ternaus/robot-surgery-segmentation/blob/master/loss.py

class LossMulti:
    """
    Loss defined as (1 - alpha) CELoss - alpha log(SoftJaccard)
    """

    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            weight = torch.from_numpy(class_weights.astype(np.float32)).cuda()
        else:
            weight = None
        # self.nll_loss = nn.NLLLoss(weight=weight)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        '''
        input: outputs n-d tensor (probability map), target tensor (0-1 map)
        '''
        loss = (1 - self.jaccard_weight) * self.ce_loss(outputs, targets)
        
        jaccard_loss = 0
        # calculate softmax first
        outputs = torch.softmax(outputs, dim=1)

        # for faster calculation
        if self.jaccard_weight > 0:
            # for each class, calculate binary loss
            for cls in range(self.num_classes):
                ious = iou_binary_torch((targets == cls).float(), outputs[:, cls])
                jaccard_loss += torch.mean(ious)
            loss -= self.jaccard_weight * torch.log(jaccard_loss / self.num_classes)
        return loss
