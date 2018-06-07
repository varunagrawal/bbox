import numpy as np
import logging


logger = logging.getLogger("bbox")

def jaccard_index(a, b):
    """
    Computes the Intersection over Union of two sets of bounding boxes.
    Also known as IoU. 
    """
    xA = np.maximum(a.x1, b.x1.T)
    yA = np.maximum(a.y1, b.y1.T)
    xB = np.minimum(a.x2, b.x2.T)
    yB = np.minimum(a.y2, b.y2.T)

    logger.debug(xA, yA, xB, yB)
    
    inter_w = xB - xA + 1
    inter_w[inter_w<0] = 0
    
    inter_h = yB - yA + 1
    inter_h[inter_h<0] = 0

    # maximum generates a (N,N) matrix which consumes a lot of memory
    # thus we are aggressive about freeing memory up.
    del(xA)
    del(yA)
    del(xB)
    del(yB)

    intersection = inter_w * inter_h 
    
    logger.debug(intersection)

    a_area = a.width * a.height
    b_area = b.width * b.height

    logger.debug(a_area, b_area)
    
    iou = intersection / (a_area + b_area.T - intersection)

    # set nan and +/- inf to 0
    iou[np.isinf(iou)] = 0
    iou[np.isnan(iou)] = 0

    return iou
