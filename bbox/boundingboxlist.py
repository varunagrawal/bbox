from copy import deepcopy
import numpy as np

from bbox.boundingbox import BoundingBox


class BoundingBoxList:
    def __init__(self, bounding_boxes:list):
        self.bboxes = np.asarray([x.numpy(two_point=True) for x in bounding_boxes])

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
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property    
    def shape(self):
        return self.bboxes.shape

    def numpy(self, two_point=False):
        if two_point:
            return self.bboxes
        else:
            bboxes = deepcopy(self.bboxes)
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
            bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
            return bboxes
    