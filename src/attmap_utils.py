import numpy as np
import torch
import torch.nn as nn

def init_attmaps(num_data, num_classes, h, w):
    attmaps = {}
    c = 1 if num_classes == 2 else num_classes
    for i in range(num_data):
        attmaps[i] = torch.rand(1, c, h, w)
    return attmaps

# calcualte prediction for next frame
def cal_attmaps(x, optflow, inverse=False):
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
    x_ = nn.MaxPool2d(3, 1, padding=1)(x_.view(1, c, h, w))

    x_ = x_.squeeze(0)
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
