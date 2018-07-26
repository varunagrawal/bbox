from copy import deepcopy
import numpy as np

from bbox.bbox2d import BBox2D


class BBox2DList:
    def __init__(self, arr, two_point=False):
        """
        Expects an iterable of bounding boxes of the form (x, y, w, h) or (x1, y1, x2, y2) if `two_point=True`.
        :param arr:
        :param two_point: Flag to indicate if `arr` is in two point format (x1, y1, x2, y2)
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
                    self.bboxes = np.asarray([BBox2D(x, two_point=two_point).numpy(two_point=True)
                                              for x in arr])

                elif all(isinstance(x, BBox2D) for x in arr):
                    # parse a list of BBox2D objects
                    self.bboxes = np.asarray(
                        [x.numpy(two_point=True) for x in arr])

                else:
                    raise Exception(
                        "Element of input is of invalid type. Elements must be all list, np.ndarray or BBox2D")

        # check if `arr` is a 2D numpy array
        elif isinstance(arr, np.ndarray):
            # remove singleton dimensions if any
            arr = np.squeeze(arr)

            # if the dimensions of the array are incorrect, raise exception.
            if arr.ndim != 2 or arr.shape[1] != 4:
                raise Exception(
                    "Invalid dimensions. Expected 2D array of size Nx4. Extra dimensions should be size 1.")

            # parse the input
            self.bboxes = np.asarray([BBox2D(x, two_point=two_point).numpy(
                two_point=True) for x in arr], dtype=np.float64)

        # if `arr` is a BBox2DList, just make a copy
        elif isinstance(arr, BBox2DList):
            self.bboxes = arr.bboxes

        else:
            raise Exception(
                "Cannot understand input type. Please use a list or a numpy array.")

    @classmethod
    def from_bbox_list(cls, bounding_boxes: list):
        """
        """
        return BBox2DList(np.asarray([x.numpy(two_point=True) for x in bounding_boxes]))

    def __eq__(self, x):
        if not isinstance(x, BBox2DList):
            return False
        return np.array_equal(self.bboxes, x.bboxes)

    def __str__(self):
        return str(self.bboxes)

    def __repr__(self):
        return str(self.bboxes)

    def __getitem__(self, key):
        return BBox2D(self.bboxes[key], two_point=True)

    def __setitem__(self, key, value):
        self.bboxes[key] = BBox2D(value).numpy(two_point=True)

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
    def w(self):
        return self.x2 - self.x1 + 1

    @property
    def height(self):
        return self.y2 - self.y1 + 1

    @property
    def h(self):
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
