"""Functions for metrics related to 2D and 3D bounding boxes."""

# pylint: disable=invalid-name,missing-docstring,assignment-from-no-return,logging-format-interpolation

import logging

import numpy as np

from bbox import BBox2D, BBox2DList, BBox3D
from bbox.geometry import polygon_area, polygon_collision, polygon_intersection

logger = logging.getLogger(__name__)


def iou_2d(a: BBox2D, b: BBox2D):
    """
    Compute the Intersection over Union (IoU) of a pair of 2D bounding boxes.

    Alias for `jaccard_index_2d`.
    """
    return jaccard_index_2d(a, b)


def jaccard_index_2d(a: BBox2D, b: BBox2D):
    """
    Compute the Jaccard Index / Intersection over Union (IoU) of a pair of 2D bounding boxes.

    Args:
        a (:py:class:`BBox2D`): 2D bounding box.
        b (:py:class:`BBox2D`): 2D bounding box.

    Returns:
        :py:class:`float`: The IoU of the 2 bounding boxes.
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

    intersection = inter_w * inter_h

    logger.debug("jaccard_index: intersection={0}".format(intersection))

    a_area = a.width * a.height
    b_area = b.width * b.height

    logger.debug("jaccard_index: a_area: {0}, b_area: {1}".format(a_area, b_area))

    iou = intersection / (a_area + b_area - intersection)

    # set nan and +/- inf to 0
    if np.isinf(iou) or np.isnan(iou):
        iou = 0

    return iou


def multi_iou_2d(a: BBox2DList, b: BBox2DList):
    """
    Compute the Intersection over Union (IoU) of two sets of 2D bounding boxes.

    Alias for `multi_jaccard_index_2d`.
    """
    return multi_jaccard_index_2d(a, b)


def multi_jaccard_index_2d(a: BBox2DList, b: BBox2DList):
    """
    Compute the Jaccard Index (Intersection over Union) of two sets of 2D bounding boxes.

    Args:
        a (:py:class:`BBox2DList`): List of 2D bounding boxes.
        b (:py:class:`BBox2DList`): List of 2D bounding boxes.

    Returns:
        :py:class:`ndarray`: IoU Matrix
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
    del xA
    del yA
    del xB
    del yB

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

def iou_3d(a: BBox3D, b: BBox3D):
    """
    Compute the Intersection over Union (IoU) of a pair of 3D bounding boxes.

    Alias for `jaccard_index_3d`.
    """
    return jaccard_index_3d(a, b)


def jaccard_index_3d(a: BBox3D, b: BBox3D):
    """
    Compute the Jaccard Index / Intersection over Union (IoU) of a pair of 3D bounding boxes.
    We compute the IoU using the top-down bird's eye view of the boxes.

    **Note**: We follow the KITTI format and assume only yaw rotations (along z-axis).

    Args:
        a (:py:class:`BBox3D`): 3D bounding box.
        b (:py:class:`BBox3D`): 3D bounding box.

    Returns:
        :py:class:`float`: The IoU of the 2 bounding boxes.
    """
    # check if the two boxes don't overlap
    if not polygon_collision(a.p[0:4, 0:2], b.p[0:4, 0:2]):
        return np.round_(0, decimals=5)

    intersection_points = polygon_intersection(a.p[0:4, 0:2], b.p[0:4, 0:2])
    inter_area = polygon_area(intersection_points)

    zmax = np.minimum(a.cz, b.cz)
    zmin = np.maximum(a.cz - a.h, b.cz - b.h)

    inter_vol = inter_area * np.maximum(0, zmax-zmin)

    a_vol = a.l * a.w * a.h
    b_vol = b.l * b.w * b.h

    union_vol = (a_vol + b_vol - inter_vol)

    iou = inter_vol / union_vol

    # set nan and +/- inf to 0
    if np.isinf(iou) or np.isnan(iou):
        iou = 0

    return np.round_(iou, decimals=5)
