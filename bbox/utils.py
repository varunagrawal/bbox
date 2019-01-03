"""
Adapted from Tomasz Malisiewicz's & Ross Girshick's code.
[https://gist.github.com/quantombone/1144423]
[https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py]
 """

import numpy as np
from bbox import BBox2D, BBox2DList
from PIL import ImageDraw


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


def aspect_ratio(bbox: BBox2D, ratios):
    """
    Enumerate box for each aspect ratio.
    """

    cx, cy = bbox.center()
    w, h = bbox.w, bbox.h
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)

    stack = np.vstack((cx - 0.5*(ws-1), cy - 0.5*(hs-1),
                       cx + 0.5*(ws-1), cy + 0.5*(hs-1)))

    boxes = BBox2DList(stack.T, two_point=True)
    return boxes


def draw_cuboid(img, p):
    draw = ImageDraw.Draw(img)
    color = tuple(np.random.choice(range(256), size=3))

    draw.line([p[0][0], p[0][1], p[1][0], p[1][1]], fill=color, width=2)
    draw.line([p[1][0], p[1][1], p[5][0], p[5][1]], fill=color, width=2)
    draw.line([p[5][0], p[5][1], p[4][0], p[4][1]], fill=color, width=2)
    draw.line([p[4][0], p[4][1], p[0][0], p[0][1]], fill=color, width=2)

    draw.line([p[3][0], p[3][1], p[2][0], p[2][1]], fill=color, width=2)
    draw.line([p[2][0], p[2][1], p[6][0], p[6][1]], fill=color, width=2)
    draw.line([p[6][0], p[6][1], p[7][0], p[7][1]], fill=color, width=2)
    draw.line([p[7][0], p[7][1], p[3][0], p[3][1]], fill=color, width=2)

    draw.line([p[0][0], p[0][1], p[3][0], p[3][1]], fill=color, width=2)
    draw.line([p[1][0], p[1][1], p[2][0], p[2][1]], fill=color, width=2)
    draw.line([p[5][0], p[5][1], p[6][0], p[6][1]], fill=color, width=2)
    draw.line([p[4][0], p[4][1], p[7][0], p[7][1]], fill=color, width=2)
    return img
