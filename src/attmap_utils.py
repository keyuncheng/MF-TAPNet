import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage


def init_attmaps_np(num_data, h, w):
    return np.zeros((num_data, h, w))

def cal_attmap_np(attmap_prev, optflow):
    '''
    Calculate Motion Flow based attention map

    input:
    attmap_prev: attention map of previous frame (stored in history)
    optflow: optical flow <prev_frame, cur_frame>
    
    return:
    attmap: Motion Flow based attention map for current frame
    '''
    h, w = optflow.shape[:2]

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    new_x = np.rint(x + optflow[:,:,0]).astype(dtype=np.int64)
    new_y = np.rint(y + optflow[:,:,1]).astype(dtype=np.int64)
    # get valid x and valid y
    new_x = np.clip(new_x, 0, w - 1)
    new_y = np.clip(new_y, 0, h - 1)

    attmap = np.zeros((h, w))
    attmap[new_y.flatten(), new_x.flatten()] = attmap_prev[y.flatten(), x.flatten()]
    
    # use the dilate operation to make attention area larger
    attmap = ndimage.grey_dilation(attmap, size=(10, 10))
    return attmap


def init_attmaps_torch(num_data, num_classes, h, w):
    attmaps = {}
    c = 1 if num_classes == 2 else num_classes
    for i in range(num_data):
        attmaps[i] = torch.rand(1, c, h, w)
    return attmaps

def cal_attmaps_torch(xs, optflows, inverse=False):
    b, c, h, w = xs.shape
    xs_ = torch.zeros(xs.shape, dtype=torch.float).cuda(non_blocking=True)
    for i in range(b):
        xs_[i] = cal_attmap_torch(xs[i], optflows[i], inverse)
    return xs_

# calcualte prediction for next frame
def cal_attmap_torch(x, optflow, inverse=False):
    c, h, w = x.shape

    if c == 1:
        x = torch.sigmoid(x).squeeze()
    else:
        x = (1 - x[0].exp()).squeeze()

    # how to do this to make attentioned area larger?
    optflow = optflow.round().long()

    cox, coy = torch.meshgrid([torch.arange(h), torch.arange(w)])

    newx = cox - optflow[1] if inverse else cox + optflow[1]
    newy = coy - optflow[0] if inverse else coy + optflow[0]

    newx = torch.clamp(newx, 0, h - 1)
    newy = torch.clamp(newy, 0, w - 1)

    x_ = torch.zeros(x.shape).cuda(non_blocking=True)
    # if c == 1:
    #     x_[newx.flatten(), newy.flatten()] = x[cox.flatten(), coy.flatten()]
    # else:
    #     for i in range(c):
    #         x_[i, newx.flatten(), newy.flatten()] = x[i, cox.flatten(), coy.flatten()]
    x_[newx.flatten(), newy.flatten()] = x[cox.flatten(), coy.flatten()]
    x_ = x_.view(1, 1, h, w)
    x_ = nn.MaxPool2d(3, 1, padding=1)(x_)

    x_ = x_.squeeze(0).float()
    return x_
    
    # output = torch.zeros(x.shape).cuda(non_blocking=True)
    # for i in range(6):
    #     if i > 0:
    #         x_ds = torch.nn.functional.interpolate(x_, scale_factor=1/(2 ** i))
    #         output += ((6.0 - i) / 15) * torch.nn.functional.interpolate(x_ds, scale_factor=2 ** i)
    #     else:
    #         output += ((6.0 - i) / 15) * x_
    # if we use prediction as attmap, this sigmoid should not be counted
    # output = torch.sigmoid(x_)
    # return output
