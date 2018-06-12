import numpy as np
from copy import deepcopy


class BBox2D:
    def __init__(self, x, two_point=False):
        # Copy constructor makes the constructor idempotent
        if isinstance(x, BBox2D):
            x = x.numpy(two_point=two_point)

        elif isinstance(x, (list, tuple)):
            if len(x) != 4:
                raise ValueError("Invalid input length. Input should have 4 elements.")
            x = np.asarray(x)

        elif isinstance(x, np.ndarray):
            if x.ndim >= 2:
                x = x.flatten()
            if x.size != 4:
                raise ValueError("Invalid input length. Input should have 4 elements.")
        else:
            raise TypeError("Expected input to constructor to be a 4 element list, tuple, numpy ndarray, or BBox2D object.")
            
        if two_point:
            w = x[2] - x[0] + 1
            h = x[3] - x[1] + 1
        else:
            w = x[2]
            h = x[3]

        self._x1 = np.float(x[0])
        self._y1 = np.float(x[1])
        self._w = np.float(w)
        self._h = np.float(h)
        
        # (x2, y2) will be used for indexing, hence we need to subtract 1
        self._x2 = self._x1 + self._w - 1
        self._y2 = self._y1 + self._h - 1

    @property
    def x1(self):
        return self._x1
    
    @x1.setter
    def x1(self, x):
        if x < 0:
            raise ValueError("Invalid input. Should be non-negative.")
        elif x > self.x2:
            raise ValueError("Value is greater than x2={0}".format(self.x2))

        self._x1 = x
        self._w = self._x2 - self._x1 + 1

    @property
    def x2(self):
        return self._x2

    @x2.setter
    def x2(self, x):
        if x < 0:
            raise ValueError("Invalid input. Should be non-negative.")
        elif x < self.x1:
            raise ValueError("Value is lesser than x1={0}".format(self.x1))
            
        self._x2 = x
        self._w = self._x2 - self._x1 + 1

    @property
    def y1(self):
        return self._y1

    @y1.setter
    def y1(self, y):
        if y < 0:
            raise ValueError("Invalid input. Should be non-negative.")
        elif y > self.y2:
            raise ValueError("Value is greater than y2={0}".format(self.y2))

        self._y1 = y
        self._h = self._y2 - self._y1 + 1

    @property
    def y2(self):
        return self._y2
        
    @y2.setter
    def y2(self, y):
        if y < 0:
            raise ValueError("Invalid input. Should be non-negative.")
        elif y < self.y1:
            raise ValueError("Value is lesser than y1={0}".format(self.y1))

        self._y2 = y
        self._h = self._y2 - self._y1 + 1

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, w):
        if w < 1:
            raise ValueError("Invalid width value. Width cannot be non-positive.")
        self._w = w
        self._x2 = self._x1 + self._w - 1

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, h):
        if h < 1:
            raise ValueError("Invalid height value. Height cannot be non-positive.")
        self._h = h
        self._y2 = self._y1 + self._h - 1

    @property
    def width(self):
        return self._w

    @width.setter
    def width(self, w):
        if w < 1:
            raise ValueError("Invalid width value. Width cannot be non-positive.")
        self._w = w
        self._x2 = self._x1 + self._w - 1

    @property
    def height(self):
        return self._h

    @height.setter
    def height(self, h):
        if h < 1:
            raise ValueError("Invalid height value. Height cannot be non-positive.")
        self._h = h
        self._y2 = self._y1 + self._h - 1

    def center(self):
        return np.array([self._x1 + self._w/2, self._y1 + self._h/2])

    def tolist(self, two_point=False):
        if two_point:
            return [self.x1, self.y1, self.x2, self.y2]
        else:
            return [self.x1, self.y1, self.w, self.h]

    def numpy(self, two_point=False):
        return np.asarray(self.tolist(two_point=two_point), dtype=np.float)

    def __repr__(self):
        return "BoundingBox(x={x}, y={y}, w={w}, h={h})".format(x=self.x1, y=self.y1, w=self.w, h=self.h)
