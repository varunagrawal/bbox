"""Bounding Box 2D list module."""

# pylint: disable=invalid-name,missing-docstring

from copy import deepcopy
import numpy as np

from bbox.bbox2d import BBox2D
from bbox.box_modes import XYXY, XYWH


class BBox2DList:
    """Bounding Box 2D list class."""

    def __init__(self, arr, mode=XYWH):
        """
        Class to reprsent a list of 2D bounding boxes.
        Expects an iterable of bounding boxes of the form
        (x, y, w, h) or (x1, y1, x2, y2) if `mode=XYXY`.

        Args:
            arr: Sequence of list/tuple/ndarray/BBox2D, each representing a single bounding box.
            mode (BoxMode): Indicator of box format (x, y, w, h) or (x1, y1, x2, y2). \
                The values are 0 for XYWH format and 1 for XYXY format.\
                    See :py:mod:`~bbox.box_modes`.

        Raises
            ValueError: If `x` is not of length 4.
            TypeError: If `x` is not of type {list, tuple, numpy.ndarray, BBox2D}

        """
        # Internally, we record the Bounding Box list as a 2D ndarray in XYXY format.

        # We convert arr to a 2D numpy array when possible
        # check if input is a list
        if isinstance(arr, list):
            # if the list is empty, set the input to be an empty numpy array
            if not arr:
                self.bboxes = np.empty((0, 4))

            # list is not empty, so we continue
            else:
                # check if the list elements are either numpy arrays or lists
                # if yes, then convert to a list of BBox2D objects
                if all(isinstance(x, np.ndarray) or isinstance(x, list) for x in arr):
                    self.bboxes = np.asarray([
                        BBox2D(x, mode=mode).numpy(mode=XYXY)
                        for x in arr])

                elif all(isinstance(x, BBox2D) for x in arr):
                    # parse a list of BBox2D objects
                    self.bboxes = np.asarray(
                        [x.numpy(mode=XYXY) for x in arr])

                else:
                    raise TypeError(
                        "Element of input is of invalid type." \
                            "Elements must be all list, np.ndarray or BBox2D")

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
                    err_msg = "Invalid dimensions. " \
                        "Expected 2D array of size Nx4 and extra dimensions should be size 1." \
                            "Got {0}".format(arr.shape)
                    raise ValueError(err_msg)

                # parse the input
                self.bboxes = np.asarray(
                    [BBox2D(x, mode=mode).numpy(mode=XYXY) for x in arr], dtype=np.float64)

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
        return repr(self.numpy())

    def __getitem__(self, key):
        return BBox2D(self.bboxes[key], mode=XYXY)

    def __setitem__(self, key, value):
        self.bboxes[key] = BBox2D(value).numpy(mode=XYXY)

    def __len__(self):
        return self.bboxes.shape[0]

    def mul(self, scale):
        """
        Scale the bounding boxes by the factor `s`.

        Args:
            scale : Scalar factor to scale by.
        """
        if not isinstance(scale, (int, float)):
            raise ValueError(
                "Bounding boxes can only be multiplied by scalar (int or float)")
        return BBox2DList(self.bboxes * scale, mode=XYXY)

    def __mul__(self, val):
        return self.mul(val)

    def __rmul__(self, val):
        return self.mul(val)

    @property
    def x1(self):
        """
        :py:class:`float`: Left x coordinate of all boxes.
        """
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
        """
        :py:class:`float`: Right x coordinate of all boxes.
        """
        return self.bboxes[:, 2]

    @x2.setter
    def x2(self, x):
        x = self._convert_attribute_input(x)
        self.bboxes[:, 2] = x

    @property
    def y1(self):
        """
        :py:class:`float`: Top y coordinate of all boxes.
        """
        return self.bboxes[:, 1]

    @y1.setter
    def y1(self, x):
        x = self._convert_attribute_input(x)
        self.bboxes[:, 1] = x

    @property
    def y2(self):
        """
        :py:class:`float`: Bottom y coordinate of all boxes.
        """
        return self.bboxes[:, 3]

    @y2.setter
    def y2(self, x):
        x = self._convert_attribute_input(x)
        self.bboxes[:, 3] = x

    @property
    def width(self):
        """
        :py:class:`float`: Width of bounding box of all boxes.
        """
        return self.x2 - self.x1 + 1

    @width.setter
    def width(self, w):
        w = self._convert_attribute_input(w)
        self.x2 = self.x1 + w - 1

    @property
    def w(self):
        """
        :py:class:`float`: Syntactic sugar for width.
        """
        return self.x2 - self.x1 + 1

    @w.setter
    def w(self, w):
        self.width = w

    @property
    def height(self):
        """
        :py:class:`float`: Height of bounding box of all boxes.
        """
        return self.y2 - self.y1 + 1

    @height.setter
    def height(self, h):
        h = self._convert_attribute_input(h)
        self.y2 = self.y1 + h - 1

    @property
    def h(self):
        """
        :py:class:`float`: Syntactic sugar for height.
        """
        return self.y2 - self.y1 + 1

    @h.setter
    def h(self, h):
        self.height = h

    @property
    def shape(self):
        """
        :py:class:`tuple`: Return the shape of the bounding boxes container in the form (N, 4).
        """
        return self.bboxes.shape

    def append(self, x, mode=XYWH):
        """
        Append a bounding box to the bounding box list.

        Args:
            x: Bounding box to append.
        """
        if isinstance(x, (tuple, list, np.ndarray)):
            try:
                x = np.asarray(x, dtype=np.float)
            except (ValueError,):
                raise ValueError(
                    "Expected list, tuple, or numpy array of ints/floats")

            if x.ndim == 1:
                x = x[np.newaxis, :]

            if x.ndim > 1 and x.shape[1] != 4:
                raise ValueError(
                    "Input should have shape Nx4, got {0}".format(x.shape))

            # Convert to BBox2D
            x = BBox2D(x, mode=mode)
            x = x.numpy(mode=XYXY).reshape(1, 4)

        elif isinstance(x, BBox2D):
            # ensure that the input is in 2 point format
            x = x.numpy(mode=XYXY).reshape(1, 4)

        elif isinstance(x, BBox2DList):
            x = x.numpy(mode=XYXY)

        else:
            raise TypeError(
                "Expected input of type (list, tuple, np.ndarray, BBox2D)")

        return BBox2DList(np.append(self.bboxes, x, axis=0), mode=XYXY)

    def insert(self, x, idx, mode=XYWH):
        """
        Insert a bounding box at a specific location.

        Args:
            x: Bounding box to insert.
            idx (:py:class:`int`): Position where to insert bounding box.
        """
        if isinstance(x, (tuple, list, np.ndarray)):
            try:
                x = np.asarray(x, dtype=np.float)
            except (ValueError,):
                raise ValueError(
                    "Expected list, tuple, or numpy array of ints/floats")

            if x.ndim > 1 or x.shape[0] != 4:
                raise ValueError(
                    "Input should have shape Nx4, got {0}".format(x.shape))

            # ensure that the input is in 2 point format
            x = BBox2D(x, mode=mode)

        elif isinstance(x, BBox2D):
            # don't need to do anything here
            pass

        else:
            raise TypeError(
                "Expected input of type (list, tuple, np.ndarray, BBox2D)")

        # ensure that the input is in 2 point format
        x = x.numpy(mode=XYXY).reshape(1, 4)

        return BBox2DList(np.insert(self.bboxes, idx, x, axis=0), mode=XYXY)

    def delete(self, index):
        """
        Delete bounding box at index from this list.

        Args:
            index (:py:class:`int`): Index of the box to delete.
        """
        return BBox2DList(np.delete(self.bboxes, index, axis=0), mode=XYXY)

    def copy(self):
        """
        Return a deep copy of this bounding box list.
        """
        return deepcopy(self)

    def numpy(self, mode=XYWH):
        """
        Return np.ndarray of shape (N, 4) representing all the bounding boxes.

        Args:
            mode (BoxMode2D): Mode in which to return the box. See :py:mod:`~bbox.box_modes`.
        """
        if mode == XYXY:
            return self.bboxes
        else:
            bboxes = deepcopy(self.bboxes)
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] + 1
            bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] + 1
            return bboxes
