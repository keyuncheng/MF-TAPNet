# Author: Keyun Cheng
# Date Created: 2019-01-01

import numpy as np
import cv2

def flow_to_arrow(flow_uv, positive=True):
    '''
    Expects a two dimensional flow image of shape [H,W,2]

    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return: flow image shown in arrow
    '''
    h, w = flow_uv.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    new_x = np.rint(x + flow_uv[:,:,0]).astype(dtype=np.int64)
    new_y = np.rint(y + flow_uv[:,:,1]).astype(dtype=np.int64)
    # clip to the boundary
    new_x = np.clip(new_x, 0, w)
    new_y = np.clip(new_y, 0, h)
    # empty image
    coords_origin = np.array([x.flatten(), y.flatten()]).T
    coords_new = np.array([new_x.flatten(), new_y.flatten()]).T

    flow_arrow = np.ones((h, w, 3), np.uint8) * 255
    for i in range(0, len(coords_origin), 1000):
        if positive:
            cv2.arrowedLine(flow_arrow, tuple(coords_origin[i]), tuple(coords_new[i]), (255, 0, 0), 2)
        else:
            cv2.arrowedLine(flow_arrow, tuple(coords_new[i]), tuple(coords_origin[i]), (255, 0, 0), 2)
    return flow_arrow
        