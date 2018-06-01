import numpy as np
from copy import deepcopy


class BoundingBox:
    def __init__(self, x, y, w, h):
        self.x1 = x
        self.y1 = y
        self.w = w
        self.h = h
        # (x2, y2) will be used for indexing, hence we need to subtract 1
        self.x2 = x + w - 1
        self.y2 = y + h - 1

    def tolist(self, two_point=False):
        if two_point:
            return [self.x1, self.y1, self.x2, self.y2]
        else:
            return [self.x1, self.y1, self.w, self.h]

    def numpy(self, two_point=False):
        return np.asarray(self.tolist(two_point), dtype=np.float)

    def __repr__(self):
        return "BoundingBox(x={x}, y={y}, w={w}, h={h})".format(x=self.x1, y=self.y1, w=self.w, h=self.h)
