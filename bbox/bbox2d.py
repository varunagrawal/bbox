import numpy as np
from copy import deepcopy


class BBox2D:
    def __init__(self, x, two_point=False):
        """
        Class to reprsent a 2D bounding box.

        Parameters
        ----------
        x
            Sequence of length 4 representing (x, y, w, h) or (x1, y1, x2, y2) depending on if `two_point` is False or True respectively.
        two_point : bool 
            Flag to indicate which format `x` is in (x, y, w, h) or (x1, y1, x2, y2).

        Attributes
        ----------
        x1 : float
            Left x coordinate
        y1 : float
            Top y coordinate
        x2 : float
            Right x coordinate
        y2 : float
            Bottom y coordinate
        width : float
            Width of bounding box
        height : float
            Height of bounding box
        w : float
            Syntactic sugar for width
        h : float
            Syntactic sugar for height

        Raises
        ------
        ValueError
            If `x` is not of length 4.
        TypeError
            If `x` is not of type {list, tuple, numpy.ndarray, BBox2D}

        """
        # Copy constructor makes the constructor idempotent
        if isinstance(x, BBox2D):
            x = x.numpy(two_point=two_point)

        elif isinstance(x, (list, tuple)):
            if len(x) != 4:
                raise ValueError(
                    "Invalid input length. Input should have 4 elements.")
            x = np.asarray(x)

        elif isinstance(x, np.ndarray):
            if x.ndim >= 2:
                x = x.flatten()
            if x.size != 4:
                raise ValueError(
                    "Invalid input length. Input should have 4 elements.")
        else:
            raise TypeError(
                "Expected input to constructor to be a 4 element list, tuple, numpy ndarray, or BBox2D object.")

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

    def __eq__(self, x):
        if not isinstance(x, BBox2D):
            return False
        return self._x1 == x.x1 \
            and self._y1 == x.y1 \
            and self._x2 == x.x2 \
            and self._y2 == x.y2

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
    def width(self):
        return self._w

    @width.setter
    def width(self, w):
        if w < 1:
            raise ValueError(
                "Invalid width value. Width cannot be non-positive.")
        self._w = w
        self._x2 = self._x1 + self._w - 1

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, w):
        self.width = w

    @property
    def height(self):
        return self._h

    @height.setter
    def height(self, h):
        if h < 1:
            raise ValueError(
                "Invalid height value. Height cannot be non-positive.")
        self._h = h
        self._y2 = self._y1 + self._h - 1

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, h):
        self.height = h

    def center(self):
        """Return center coordinates of the bounding box"""
        return np.array([self._x1 + (self._w-1)/2, self._y1 + (self._h-1)/2])

    def aspect_ratio(self, ratio):
        """
        Return 2D BBox mapped to new aspect ratio denoted by `ratio`.
        The ratio should be given as the result of `height / width`
        """
        area = self.h * self.w
        area_ratio = area / ratio
        new_width = np.round(np.sqrt(area_ratio))
        new_height = np.round(ratio * new_width)
        new_bbox = BBox2D((self.x1, self.y1, new_width, new_height),
                          two_point=False)
        return new_bbox

    def tolist(self, two_point=False):
        """
        Return bounding box as a `list` of 4 numbers. 
        Format depends on `two_point` flag.
        """
        if two_point:
            return [self.x1, self.y1, self.x2, self.y2]
        else:
            return [self.x1, self.y1, self.w, self.h]

    def copy(self):
        return deepcopy(self)

    def numpy(self, two_point=False):
        """
        Return bounding box as a numpy vector of length 4.
        Format depends on `two_point` flag.
        """
        return np.asarray(self.tolist(two_point=two_point), dtype=np.float)

    def __repr__(self):
        return "BBox2D([{x}, {y}, {w}, {h}])".format(x=self.x1, y=self.y1, w=self.w, h=self.h)

    def mul(self, s):
        if not isinstance(s, (int, float)):
            raise ValueError(
                "Bounding boxes can only be multiplied by scalar (int or float)")
        return BBox2D([self.x1*s, self.y1*s, self.x2*s, self.y2*s], two_point=True)

    def __mul__(self, s):
        return self.mul(s)

    def __rmul__(self, s):
        return self.mul(s)
