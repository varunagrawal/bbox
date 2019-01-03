from copy import deepcopy
import numpy as np

from bbox.bbox2d import BBox2D


class BBox2DList:
    def __init__(self, arr, two_point=False):
        """
        Class to reprsent a list of 2D bounding boxes.
        Expects an iterable of bounding boxes of the form (x, y, w, h) or (x1, y1, x2, y2) if `two_point=True`.

        Parameters
        ----------
        arr:
            Sequence of list/tuple/ndarray/BBox2D, each representing a single bounding box.
        two_point : bool
            Flag to indicate which format `x` is in (x, y, w, h) or (x1, y1, x2, y2).

        Attributes
        ----------
        x1 : float
            Left x coordinate of all boxes
        y1 : float
            Top y coordinate of all boxes
        x2 : float
            Right x coordinate of all boxes
        y2 : float
            Bottom y coordinate of all boxes
        width : float
            Width of bounding box of all boxes
        height : float
            Height of bounding box of all boxes
        w : float
            Syntactic sugar for width
        h : float
            Syntactic sugar for height
        shape : np.ndarray
            Return the shape of the bounding boxes container in the form (N, 4).

        Raises
        ------
        ValueError
            If `x` is not of length 4.
        TypeError
            If `x` is not of type {list, tuple, numpy.ndarray, BBox2D}

        """
        # Internally, we record the Bounding Box list as a 2D ndarray in two_point format.

        # We convert arr to a 2D numpy array when possible
        # check if input is a list
        if isinstance(arr, list):
            # if the list is empty, set the input to be an empty numpy array
            if len(arr) == 0:
                self.bboxes = np.empty((0, 4))

            # list is not empty, so we continue
            else:
                # check if the list elements are either numpy arrays or lists
                # if yes, then convert to a list of BBox2D objects
                if all(isinstance(x, np.ndarray) or isinstance(x, list) for x in arr):
                    self.bboxes = np.asarray([
                        BBox2D(x, two_point=two_point).numpy(two_point=True)
                        for x in arr])

                elif all(isinstance(x, BBox2D) for x in arr):
                    # parse a list of BBox2D objects
                    self.bboxes = np.asarray(
                        [x.numpy(two_point=True) for x in arr])

                else:
                    raise TypeError(
                        "Element of input is of invalid type. Elements must be all list, np.ndarray or BBox2D")

        # check if `arr` is a 2D numpy array
        elif isinstance(arr, np.ndarray):
            # Check for empty ndarray
            if arr.ndim == 2 and arr.shape[0] == 0:
                self.bboxes = np.empty((0, 4))

            else:
                # if input is a 1D vector, we add the second dimension
                if arr.ndim == 1 and arr.shape[0] == 4:
                    arr = arr[np.newaxis, :]

                # if the dimensions of the array are incorrect, raise exception.
                if arr.ndim != 2 or arr.shape[1] != 4:
                    raise ValueError(
                        "Invalid dimensions. Expected 2D array of size Nx4. Extra dimensions should be size 1. Got {0}".format(arr.shape))

                # parse the input
                self.bboxes = np.asarray(
                    [BBox2D(x, two_point=two_point).numpy(two_point=True) for x in arr], dtype=np.float64)

        # if `arr` is a BBox2DList, just make a copy
        elif isinstance(arr, BBox2DList):
            self.bboxes = arr.bboxes

        else:
            raise TypeError(
                "Invalid input type. Please use a list or a numpy array.")

    def __eq__(self, x):
        if not isinstance(x, BBox2DList):
            return False
        return np.array_equal(self.bboxes, x.bboxes)

    def __str__(self):
        return str(self.numpy())

    def __repr__(self):
        return str(self.numpy())

    def __getitem__(self, key):
        return BBox2D(self.bboxes[key], two_point=True)

    def __setitem__(self, key, value):
        self.bboxes[key] = BBox2D(value).numpy(two_point=True)

    def __len__(self):
        return self.bboxes.shape[0]

    def mul(self, s):
        if not isinstance(s, (int, float)):
            raise ValueError(
                "Bounding boxes can only be multiplied by scalar (int or float)")
        return BBox2DList(self.bboxes * s, two_point=True)

    def __mul__(self, s):
        return self.mul(s)

    def __rmul__(self, s):
        return self.mul(s)

    @property
    def x1(self):
        return self.bboxes[:, 0]

    def _convert_attribute_input(self, x):
        if not isinstance(x, (list, tuple, np.ndarray)):
            raise TypeError("Input should be of type list/tuple/ndarray")
        x = np.asarray(x)
        if x.ndim != 1 or (x.ndim == 1 and x.shape[0] != self.bboxes.shape[0]):
            raise ValueError("Invalid shape of input. Expected shape to be ({0},)".format(
                self.bboxes.shape[0]))
        return x

    @x1.setter
    def x1(self, x):
        x = self._convert_attribute_input(x)
        self.bboxes[:, 0] = x

    @property
    def x2(self):
        return self.bboxes[:, 2]

    @x2.setter
    def x2(self, x):
        x = self._convert_attribute_input(x)
        self.bboxes[:, 2] = x

    @property
    def y1(self):
        return self.bboxes[:, 1]

    @y1.setter
    def y1(self, x):
        x = self._convert_attribute_input(x)
        self.bboxes[:, 1] = x

    @property
    def y2(self):
        return self.bboxes[:, 3]

    @y2.setter
    def y2(self, x):
        x = self._convert_attribute_input(x)
        self.bboxes[:, 3] = x

    @property
    def width(self):
        return self.x2 - self.x1 + 1

    @width.setter
    def width(self, w):
        w = self._convert_attribute_input(w)
        self.x2 = self.x1 + w - 1

    @property
    def w(self):
        return self.x2 - self.x1 + 1

    @w.setter
    def w(self, w):
        self.width = w

    @property
    def height(self):
        return self.y2 - self.y1 + 1

    @height.setter
    def height(self, h):
        h = self._convert_attribute_input(h)
        self.y2 = self.y1 + h - 1

    @property
    def h(self):
        return self.y2 - self.y1 + 1

    @h.setter
    def h(self, h):
        self.height = h

    @property
    def shape(self):
        return self.bboxes.shape

    def append(self, x, two_point=False):
        if isinstance(x, (tuple, list, np.ndarray)):
            try:
                x = np.asarray(x)
            except:
                raise ValueError("Expected numpy array or list")

            if x.ndim == 1:
                x = x[np.newaxis, :]

            if x.ndim > 1 and x.shape[1] != 4:
                raise ValueError(
                    "Input should have shape Nx4, got {0}".format(x.shape))

            # Convert to BBox2D
            x = BBox2D(x, two_point=two_point)
            x = x.numpy(two_point=True).reshape(1, 4)

        elif isinstance(x, BBox2D):
            # ensure that the input is in 2 point format
            x = x.numpy(two_point=True).reshape(1, 4)

        elif isinstance(x, BBox2DList):
            x = x.numpy(two_point=True)

        else:
            raise TypeError(
                "Expected input of type (list, tuple, np.ndarray, BBox2D)")

        return BBox2DList(np.append(self.bboxes, x, axis=0), two_point=True)

    def insert(self, x, idx, two_point=False):
        if isinstance(x, (tuple, list, np.ndarray)):
            try:
                x = np.asarray(x)
            except:
                raise ValueError("Expected numpy array or list")

            if x.ndim > 1 or x.shape[0] != 4:
                raise ValueError(
                    "Input should have shape Nx4, got {0}".format(x.shape))

            # ensure that the input is in 2 point format
            x = BBox2D(x, two_point=two_point)

        elif isinstance(x, BBox2D):
            pass

        else:
            raise TypeError(
                "Expected input of type (list, tuple, np.ndarray, BBox2D)")

        # ensure that the input is in 2 point format
        x = x.numpy(two_point=True).reshape(1, 4)

        return BBox2DList(np.insert(self.bboxes, idx, x, axis=0), two_point=True)

    def delete(self, idx):
        return BBox2DList(np.delete(self.bboxes, idx, axis=0), two_point=True)

    def copy(self):
        return deepcopy(self)

    def numpy(self, two_point=False):
        """Return np.ndarray of shape (N, 4) representing all the bounding boxes"""
        if two_point:
            return self.bboxes
        else:
            bboxes = deepcopy(self.bboxes)
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] + 1
            bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] + 1
            return bboxes
