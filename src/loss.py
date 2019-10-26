import torch
from torch import nn
import numpy as np
import tqdm

from metrics import (
    iou_binary_torch,
    dice_binary_torch,
    iou_multi_np,
    dice_multi_np,
    )

from attmap_utils import cal_attmaps_torch


# parts of implementation are borrowed from:
# https://github.com/ternaus/robot-surgery-segmentation/blob/master/loss.py

class LossMulti:
    """
    Loss = (1 - jaccard_weight) * CELoss - jaccard_weight * log(JaccardLoss)
    """

    def __init__(self, num_classes, jaccard_weight, class_weights=None):
        if class_weights is not None:
            weight = torch.from_numpy(class_weights.astype(np.float32)).cuda()
        else:
            weight = None

        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        '''
        @param outputs: n-d tensor (probability map)
        @param targets: tensor (0-1 map)

        @return loss: loss
        '''

        # empty input

        loss = (1 - self.jaccard_weight) * self.ce_loss(outputs, targets)
        
        jaccard_loss = 0
        # turn logits into prob maps
        outputs = torch.softmax(outputs, dim=1)

        if self.jaccard_weight > 0:
            # for each class, calculate jaccard loss
            for cls in range(self.num_classes):
                metric = iou_binary_torch((targets == cls).float(), outputs[:, cls])
                # metric = iou_binary_dice((targets == cls).float(), outputs[:, cls:cls+1])
                
                # mean of metric(class = cls) for all inputs 
                jaccard_loss += torch.mean(metric)
            loss -= self.jaccard_weight * torch.log(jaccard_loss / self.num_classes)
        return loss


class LossMultiSemi(object):
    """multi class loss for semi-supervised problem"""
    def __init__(self, num_classes, jaccard_weight, alpha, semi_method, class_weights=None):
        super(LossMultiSemi, self).__init__()
        self.loss_multi = LossMulti(num_classes, jaccard_weight, class_weights)
        self.semi_method = semi_method
        self.alpha = alpha

    def __call__(self, outputs, targets, **kwargs):
        # TODO: intergrate more choices of semi_percentage in semi-supervised learning
        # labeled outputs and targets
        labeled = kwargs['labeled']
        l_outputs = outputs[labeled == True]
        l_targets = targets[labeled == True]

        if self.semi_method == 'ignore':
            loss = self.loss_multi(l_outputs, l_targets)
        elif self.semi_method == 'aug_gt':
            # like fully supervised loss
            loss = self.loss_multi(outputs, targets)
        elif self.semi_method == 'rev_flow':
            ul_outputs = outputs[labeled == False]
            # ul_targets are targets of previous labeled inputs
            ul_targets = targets[labeled == False]
            ul_optflows = kwargs['optflow'][labeled == False]
            # inverse outputs
            inv_outputs = cal_attmaps_torch(ul_outputs, ul_optflows, inverse=True)
            if self.alpha is not None:
                loss = None
                if l_targets.shape[0] > 0:
                    loss = (1 - self.alpha) * self.loss_multi(l_outputs, l_targets)
                if ul_targets.shape[0] > 0:
                    un_loss = self.alpha * self.loss_multi(inv_outputs, ul_targets)
                    if loss is None:
                        loss = un_loss
                    else:
                        loss += un_loss
            else:
                '''
                EXPERIMENT: 
                attention map will be updated for every data
                but only do backward pass for supervised data
                '''
                loss = self.loss_multi(l_outputs, l_targets) 

        return loss