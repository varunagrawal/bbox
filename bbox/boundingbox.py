import numpy as np
from copy import deepcopy


class BoundingBox:
    def __init__(self, x, two_point=False):
        # Copy constructor makes the constructor idempotent
        if isinstance(x, BoundingBox):
            x = x.numpy(two_point=two_point)

        elif isinstance(x, list):
            x = np.asarray(x)

        elif isinstance(x, np.ndarray):
            if x.ndim >= 2:
                x = x.flatten()
            if x.size != 4:
                raise Exception("Invalid input length. Input should have 4 elements.")

        if two_point:
            x[2] = x[2] - x[0] + 1
            x[3] = x[3] - x[1] + 1

        self.x1 = np.float(x[0])
        self.y1 = np.float(x[1])
        self.w = np.float(x[2])
        self.h = np.float(x[3])
        
        # (x2, y2) will be used for indexing, hence we need to subtract 1
        self.x2 = self.x1 + self.w - 1
        self.y2 = self.y1 + self.h - 1

    # TODO 
    # Set up setters and getters

    def tolist(self, two_point=False):
        if two_point:
            return [self.x1, self.y1, self.x2, self.y2]
        else:
            return [self.x1, self.y1, self.w, self.h]

    def numpy(self, two_point=False):
        return np.asarray(self.tolist(two_point), dtype=np.float)

    def __repr__(self):
        return "BoundingBox(x={x}, y={y}, w={w}, h={h})".format(x=self.x1, y=self.y1, w=self.w, h=self.h)

    def __str__(self):
        return "(x={x}, y={y}, w={w}, h={h})".format(x=self.x1, y=self.y1, w=self.w, h=self.h)