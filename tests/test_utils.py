import numpy as np
from bbox import BBox2D, BBox2DList, BBox3D
from bbox.box_modes import XYXY, XYWH
from bbox.utils import nms, aspect_ratio
from PIL import Image
import pytest


def naive_nms(dets, thresh):
    """
    Courtesy of Ross Girshick
    [https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py]
    """
    if dets.shape[0] == 0:
        return np.array([]).astype(np.int)

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep).astype(np.int)


def test_blank_nms():
    bbl = BBox2DList([])
    scores = np.random.rand(40)
    thresh = 0.0001
    keep = nms(bbl, scores, thresh)
    naive_keep = naive_nms(bbl.numpy(), thresh)
    assert np.all(keep == naive_keep)


@pytest.mark.filterwarnings("ignore:.*[double_scalars|true_divide|less_equal]")
def test_nms():
    # bbl, scores, thresh
    np.random.seed(529)
    x1 = np.random.randint(0, 50, size=40)
    y1 = np.random.randint(0, 50, size=40)
    w = np.random.randint(0, 50, size=40)
    h = np.random.randint(0, 50, size=40)
    x2 = x1 + w - 1
    y2 = y1 + h - 1

    bboxes_list = np.stack((x1, y1, x2, y2), axis=1)
    scores = np.random.rand(40)
    thresh = 0.0001

    dets = np.hstack((bboxes_list, scores[:, np.newaxis]))
    naive_keep = naive_nms(dets, thresh)

    # see how easy it is with `bbox`
    bblist = np.stack((x1, y1, w, h), axis=1)
    bbl = BBox2DList(bblist)
    keep = nms(bbl, scores, thresh)
    assert np.all(keep == naive_keep)


def test_aspect_ratio():
    box = BBox2D([0, 0, 15, 15], mode=XYXY)
    box_ar = aspect_ratio(box, [0.5, 1, 2])
    gt_box_ar = BBox2DList(np.array([[-3.5,  2., 18.5, 13.],
                                     [0.,  0., 15., 15.],
                                     [2.5, -3, 12.5, 18.]]),
                           mode=XYXY)

    assert box_ar == gt_box_ar
