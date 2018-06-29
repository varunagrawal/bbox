import numpy as np
import logging
import warnings

from bbox import BBox2D, BBox2DList, BBox3D


logger = logging.getLogger(__name__)


def jaccard_index_2d(a, b):
    """
    Computes the Intersection over Union (IoU) of 2 2-d bounding boxes.
    """
    xA = np.maximum(a.x1, b.x1)
    yA = np.maximum(a.y1, b.y1)
    xB = np.minimum(a.x2, b.x2)
    yB = np.minimum(a.y2, b.y2)

    logger.debug("xA={0} yA={1} xB={2} yB={3}".format(xA, yA, xB, yB))

    inter_w = xB - xA + 1
    inter_w = inter_w * (inter_w >= 0)

    inter_h = yB - yA + 1
    inter_h = inter_h * (inter_h >= 0)

    # maximum generates a (N,N) matrix which consumes a lot of memory
    # thus we are aggressive about freeing memory up.
    del(xA)
    del(yA)
    del(xB)
    del(yB)

    intersection = inter_w * inter_h

    logger.debug("jaccard_index: intersection={0}".format(intersection))

    a_area = a.width * a.height
    b_area = b.width * b.height

    logger.debug(
        "jaccard_index: a_area: {0}, b_area: {1}".format(a_area, b_area))

    iou = intersection / (a_area + b_area - intersection)

    # set nan and +/- inf to 0
    if np.isinf(iou) or np.isnan(iou):
        iou = 0

    return iou


def multi_jaccard_index_2d(a, b):
    """
    Computes the Intersection over Union of two sets of bounding boxes.
    Also known as IoU. 
    """
    # We need to add a trailing dimension so that max/min gives us a (N,N) matrix
    xA = np.maximum(a.x1[:, np.newaxis], b.x1[:, np.newaxis].T)
    yA = np.maximum(a.y1[:, np.newaxis], b.y1[:, np.newaxis].T)
    xB = np.minimum(a.x2[:, np.newaxis], b.x2[:, np.newaxis].T)
    yB = np.minimum(a.y2[:, np.newaxis], b.y2[:, np.newaxis].T)

    logger.debug("\nmulti_jaccard_index:\nxA\n{0}\nyA\n{1}\nxB\n{2}\nyB\n{3}".format(
        xA, yA, xB, yB))

    inter_w = xB - xA + 1
    inter_w[inter_w < 0] = 0

    inter_h = yB - yA + 1
    inter_h[inter_h < 0] = 0

    # maximum generates a (N,N) matrix which consumes a lot of memory
    # thus we are aggressive about freeing memory up.
    del(xA)
    del(yA)
    del(xB)
    del(yB)

    intersection = inter_w * inter_h
    logger.debug(
        "\nmulti_jaccard_index intersection:\n {0}".format(intersection))

    a_area = a.width[:, np.newaxis] * a.height[:, np.newaxis]
    b_area = b.width[:, np.newaxis] * b.height[:, np.newaxis]
    logger.debug(
        "\nmulti_jaccard_index:\n a_area:\n {0} \nb_area:\n {1}".format(a_area, b_area))

    iou = intersection / (a_area + b_area.T - intersection)

    # set nan and +/- inf to 0
    iou[np.isinf(iou)] = 0
    iou[np.isnan(iou)] = 0

    return iou


def jaccard_index_3d(a: BBox3D, b: BBox3D):
    warnings.warn(
        "3D IoU is incomplete and incorrect for non-axis-aligned cuboids")

    x1 = np.maximum(a.p1[0], b.p1[0])
    y1 = np.maximum(a.p1[1], b.p1[1])
    z1 = np.maximum(a.p1[2], b.p1[2])
    x2 = np.minimum(a.p7[0], b.p7[0])
    y2 = np.minimum(a.p7[1], b.p7[1])
    z2 = np.minimum(a.p7[2], b.p7[2])

    il = np.linalg.norm(x1-x2)
    il = il * (il >= 0)
    iw = np.linalg.norm(y1-y2)
    iw = iw * (iw >= 0)
    ih = np.linalg.norm(z1-z2)
    ih = ih * (ih >= 0)
    # print(il, iw, ih)

    inter_vol = il * iw * ih

    a_vol = a.l * a.w * a.h
    b_vol = b.l * b.w * b.h

    union_vol = (a_vol + b_vol - inter_vol)

    iou = inter_vol / union_vol

    # set nan and +/- inf to 0
    if np.isinf(iou) or np.isnan(iou):
        iou = 0

    return iou
