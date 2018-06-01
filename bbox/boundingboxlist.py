from copy import deepcopy
import numpy as np

from bbox.boundingbox import BoundingBox


class BoundingBoxList:
    def __init__(self, arr):
        """"""
        if arr.shape[0] > 0 and arr.shape[1] != 4:
            raise Exception("Invalid bounding box dimension. Expected list of array of size 4.")
        self.bboxes = np.asarray(arr, dtype=np.float64)

    @classmethod
    def from_bbox_list(cls, bounding_boxes:list):
        """
        """
        return BoundingBoxList(np.asarray([x.numpy(two_point=True) for x in bounding_boxes]))

    def __str__(self):
        return str(self.bboxes)

    def __repr__(self):
        return str(self.bboxes)

    def __getitem__(self, key):
        return self.bboxes[key]

    def __len__(self):
        return self.bboxes.shape[0]

    @property
    def x1(self):
        return self.bboxes[:, 0]
    
    @x1.setter
    def x1(self, x):
        if isinstance(x, list):
            x = np.asarray(x)
        self.bboxes[:, 0] = x

    @property
    def x2(self):
        return self.bboxes[:, 2]

    @property
    def y1(self):
        return self.bboxes[:, 1]
    
    @property
    def y2(self):
        return self.bboxes[:, 3]

    @property
    def width(self):
        return self.x2 - self.x1 + 1

    @property
    def height(self):
        return self.y2 - self.y1 + 1

    @property    
    def shape(self):
        return self.bboxes.shape

    def numpy(self, two_point=False):
        if two_point:
            return self.bboxes
        else:
            bboxes = deepcopy(self.bboxes)
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] + 1
            bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] + 1
            return bboxes
    