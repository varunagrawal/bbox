import numpy as np
import pytest
import logging
import pendulum

# needed to ignore matplotlib warnings
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


from bbox import BBox2D, BBox2DList, BBox3D
from bbox.metrics import jaccard_index_2d, multi_jaccard_index_2d, jaccard_index_3d

logger = logging.getLogger("test_metrics")


def naive_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def test_single_jaccard_index_2d():
    # sample bounxing boxes (x, y, w, h)
    bbox1 = [39, 63, 203, 112]
    bbox2 = [54, 66, 198, 114]

    a = BBox2D(bbox1)
    b = BBox2D(bbox2)
    iou = jaccard_index_2d(a, b)

    bbox1[2], bbox1[3] = bbox1[2] + bbox1[0], bbox1[3] + bbox1[1]
    bbox2[2], bbox2[3] = bbox2[2] + bbox2[0], bbox2[3] + bbox2[1]

    gt_iou = naive_intersection_over_union(bbox1, bbox2)

    logger.debug("IoU={0}, Naive IoU={1}".format(iou, gt_iou))
    assert iou == gt_iou


@pytest.mark.filterwarnings("ignore:.*double_scalars")
def test_invalid_jaccard_index_2d():
    # sample bounxing boxes (x, y, w, h)
    # should result in `nan` and be corrected to 0
    bbox1 = [39, 63, 0, 0]
    bbox2 = [54, 66, 0, 0]

    a = BBox2D(bbox1)
    b = BBox2D(bbox2)
    iou = jaccard_index_2d(a, b)

    logger.debug("IoU={0}".format(iou))
    assert iou == 0


def test_multi_jaccard_index_2d():
    # bounding boxes of the form (x, y, w, h)
    bboxes_1 = [[39, 63, 203, 112], [49, 75, 203, 125], [
        31, 69, 201, 125], [50, 72, 197, 121], [35, 51, 196, 110]]
    bboxes_2 = [[54, 66, 198, 114], [42, 78, 186, 126], [
        18, 63, 235, 135], [54, 72, 198, 120], [36, 60, 180, 108]]

    # generate the BBox2DLists
    a = BBox2DList(bboxes_1)
    b = BBox2DList(bboxes_2)

    # Our method
    iou = multi_jaccard_index_2d(a, b)

    # generate IoU matrix using naive implementation
    gt_iou = np.zeros((len(bboxes_1), len(bboxes_2)))
    for i, x in enumerate(bboxes_1):
        for j, y in enumerate(bboxes_2):
            bx = [x[0], x[1], x[2]+x[0], x[3]+x[1]]
            by = [y[0], y[1], y[2]+y[0], y[3]+y[1]]
            gt_iou[i, j] = naive_intersection_over_union(bx, by)

    assert gt_iou.shape == iou.shape
    assert np.array_equal(gt_iou, iou)


@pytest.mark.filterwarnings("ignore:.*true_divide")
def test_multi_jaccard_index_2d_performance():
    """
    Test the performance of `multi_jaccard_index_2d` on 5,000 randomly sampled bounding boxes.
    """
    # sample bounding boxes and create BBox2DList
    bboxes = np.random.randint(low=0, high=500, size=(5000, 4))
    bbl = BBox2DList(bboxes)

    # time the performance
    start = pendulum.now()
    _ = multi_jaccard_index_2d(bbl, bbl)
    dt = pendulum.now() - start

    # our runtime should be less than 3 seconds for 5k boxes
    assert dt.microseconds < 1e6
    assert dt.seconds < 3


def test_jaccard_index_3d_identity():
    bb = BBox3D(3.163, z=2.468, y=34.677, height=1.529, width=1.587, length=3.948,
                rw=0.7002847660410397, rx=-0, ry=-0, rz=-0.7138636049350369)
    assert jaccard_index_3d(bb, bb) == 1


def test_jaccard_index_3d_rotation_only():
    """
    Since we take the data from KITTI, we need to swap the Y and Z axes.
    """
    a = BBox3D(3.163, z=2.468, y=34.677, height=1.529, width=1.587, length=3.948,
               euler_angles=[0, 0, -1.59])
    b = BBox3D(3.163, z=2.468, y=34.677, height=1.529, width=1.587, length=3.948,
               euler_angles=[0, 0, -1.2])

    assert jaccard_index_3d(a, b) == 0.62952


def test_jaccard_index_3d_rotation_height():
    a = BBox3D(3.163, z=2.468, y=34.677, height=1.529, width=1.587, length=3.948,
               rw=0.7002847660410397, rx=-0, ry=-0, rz=-0.7138636049350369)
    b = BBox3D(3.163, z=1.468, y=34.677, height=1.529, width=1.587, length=3.948,
               rw=0.8253356149096783, rx=-0, ry=-0, rz=-0.5646424733950354)

    assert jaccard_index_3d(a, b) == 0.15428


def test_jaccard_index_3d():
    a = BBox3D(x=3.163, z=2.468, y=34.677, height=1.529, width=1.587, length=3.948,
               rw=0.7002847660410397, rx=-0, ry=-0, rz=-0.7138636049350369)
    b = BBox3D(x=3.18, z=2.27, y=34.38, height=1.41, width=1.58, length=4.36,
               rx=-0, ry=-0, rz=-0.7103532724176078, rw=0.7038453156522361)

    assert jaccard_index_3d(a, b) == 0.71232


def test_jaccard_index_3d_euler_angles():
    a = BBox3D(3.163, z=2.468, y=34.677, height=1.529, width=1.587, length=3.948,
               euler_angles=[0, 0, 1.59])
    b = BBox3D(3.18, z=2.27, y=34.38, height=1.41, width=1.58, length=4.36,
               euler_angles=[0, 0, 1.58])

    assert jaccard_index_3d(a, b) == 0.71636

def test_jaccard_index_3d_polygon_collision():
    # these two boxes should not overlap
    a = BBox3D(x=3.163, z=2.468, y=34.677, height=1.529, width=1.587, length=3.948,
               rw=0.7002847660410397, rx=-0, ry=-0, rz=-0.7138636049350369)
    b = BBox3D(x=30.18, z=20.27, y=90.38, height=1.41, width=1.58, length=4.36,
               rx=-0, ry=-0, rz=-0.7103532724176078, rw=0.7038453156522361)

    assert jaccard_index_3d(a, b) == 0

@pytest.mark.filterwarnings("ignore:.*double_scalars")
def test_invalid_jaccard_index_3d():
    # H=0,W=0,L=0, so the IoU should be nan
    bb = BBox3D(3.163, z=2.468, y=34.677, height=0, width=0, length=0,
                rw=0.7002847660410397, rx=-0, ry=-0, rz=-0.7138636049350369)
    assert jaccard_index_3d(bb, bb) == 0


def visualize_boxes(box_list):
    for b in box_list:
        polygon = Polygon(b.p[0:4, 0:2], fill=False)
        plt.gca().add_patch(polygon)

    # inter_points = []
    # for p in inter_points:
    #     plt.plot(p[0], p[1], 'bx')

    plt.axis('scaled')
    plt.show()
