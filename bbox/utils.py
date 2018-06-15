"""
Adapted from Tomasz Malisiewicz's & Ross Girshick's code.
[https://gist.github.com/quantombone/1144423]
[https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py]
 """

import numpy as np


def nms(bbl, scores, thresh):
    """
    Perform fast non-maximum suppression on a set of bounding boxes given their associated confidences.
    """
    if bbl.shape[0] == 0:
        return np.array([]).astype(np.int)

    areas = bbl.w * bbl.h
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(bbl.x1[i], bbl.x1[order[1:]])
        yy1 = np.maximum(bbl.y1[i], bbl.y1[order[1:]])
        xx2 = np.minimum(bbl.x2[i], bbl.x2[order[1:]])
        yy2 = np.minimum(bbl.y2[i], bbl.y2[order[1:]])
    
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        overlap = inter / (areas[i] + areas[order[1:]] - inter)

        idx = np.where(overlap <= thresh)[0]
        order = order[idx + 1]

    return np.array(keep).astype(np.int)
