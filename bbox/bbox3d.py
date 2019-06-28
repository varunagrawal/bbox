"""3D bounding box module."""

# pylint: disable=invalid-name,missing-docstring

from copy import deepcopy

import numpy as np
from pyquaternion import Quaternion


class BBox3D:
    """
    Class for 3D Bounding Boxes (3-orthotope).
    It takes either the center of the 3D bounding box or the back-bottom-left corner, \
        the width, height and length of the box, and quaternion values to indicate the rotation.

    Args:
        x (:py:class:`float`): X axis coordinate of 3D bounding box. \
            Can be either center of bounding box or back-bottom-left corner.
        y (:py:class:`float`): Y axis coordinate of 3D bounding box. \
            Can be either center of bounding box or back-bottom-left corner.
        z (:py:class:`float`): Z axis coordinate of 3D bounding box. \
            Can be either center of bounding box or back-bottom-left corner.
        length (:py:class:`float`, optional): The length of the box (default is 1).
        width (:py:class:`float`, optional): The width of the box (default is 1).
        height (:py:class:`float`, optional): The height of the box (default is 1).
        rw (:py:class:`float`, optional): The real part of the rotation quaternion (default is 1).
        rx (:py:class:`int`, optional): The first element of the quaternion vector (default is 0).
        ry (:py:class:`int`, optional): The second element of the quaternion vector (default is 0).
        rz (:py:class:`int`, optional): The third element of the quaternion vector (default is 0).
        euler_angles (:py:class:`list` or :py:class:`ndarray` of float, optional): Sequence of \
            euler angles in `[x, y, z]` rotation order (the default is None).
        is_center (`bool`, optional): Flag to indicate if the provided coordinate is the \
            center of the box (the default is True).
    """

    def __init__(self, x, y, z,
                 length=1, width=1, height=1,
                 rw=1, rx=0, ry=0, rz=0, q=None,
                 euler_angles=None, is_center=True):
        if is_center:
            self._cx, self._cy, self._cz = x, y, z
            self._c = np.array([x, y, z])
        else:
            self._cx = x + length/2
            self._cy = y + width/2
            self._cz = z + height/2
            self._c = np.array([self._cx, self._cy, self._cz])

        self._w, self._h, self._l = width, height, length

        if euler_angles:
            # we need to apply y, z and x rotations in order
            # http://www.euclideanspace.com/maths/geometry/rotations/euler/index.htm
            self._q = Quaternion(axis=[0, 1, 0], angle=euler_angles[1]) * \
                Quaternion(axis=[0, 0, 1], angle=euler_angles[2]) * \
                Quaternion(axis=[1, 0, 0], angle=euler_angles[0])

        elif q is not None:
            self._q = Quaternion(q)
        else:
            self._q = Quaternion(rw, rx, ry, rz)

    @property
    def center(self):
        """
        Attribute to access center coordinates of box in (x, y, z) format.
        Can be set to :py:class:`list` or :py:class:`ndarray` of float.

        Returns:
            :py:class:`ndarray` of float: 3-dimensional vector representing (x, y, z) coordinates \
                of the box.

        Raises:
            ValueError: If `c` is not a vector/list of length 3.
        """
        return self._c

    @center.setter
    def center(self, c):
        if len(c) != 3:
            raise ValueError("Center coordinates should be a vector of size 3")
        self._c = c

    def __valid_scalar(self, x):
        if not np.isscalar(x):
            raise ValueError("Value should be a scalar")
        else:  # x is a scalar so we check for numeric type
            if not isinstance(x, (np.float, np.int)):
                raise TypeError("Value needs to be either a float or an int")
        return x

    @property
    def cx(self):
        """
        :py:class:`float`: X coordinate of center.
        """
        return self._cx

    @cx.setter
    def cx(self, x):
        self._cx = self.__valid_scalar(x)

    @property
    def cy(self):
        """
        :py:class:`float`: Y coordinate of center.
        """
        return self._cy

    @cy.setter
    def cy(self, x):
        self._cy = self.__valid_scalar(x)

    @property
    def cz(self):
        """
        :py:class:`float`: Z coordinate of center.
        """
        return self._cz

    @cz.setter
    def cz(self, x):
        self._cz = self.__valid_scalar(x)

    @property
    def q(self):
        """
        Syntactic sugar for the rotation quaternion of the box.

        Returns
            :py:class:`ndarray` of float: Quaternion values in (w, x, y, z) form.
        """
        return np.hstack((self._q.real, self._q.imaginary))

    @q.setter
    def q(self, q):
        if not isinstance(q, (list, tuple, np.ndarray, Quaternion)):
            raise TypeError(
                "Value shoud be either list, numpy array or Quaterion")
        if isinstance(q, (list, tuple, np.ndarray)) and len(q) != 4:
            raise ValueError("Quaternion input should be a vector of size 4")

        self._q = Quaternion(q)

    @property
    def quaternion(self):
        """
        The rotation quaternion.

        Returns:
            :py:class:`ndarray` of float: Quaternion values in (w, x, y, z) form.
        """
        return self.q

    @quaternion.setter
    def quaternion(self, q):
        self.q = q

    @property
    def l(self):
        """
        :py:class:`float`: Syntactic sugar for length of the box.
        """
        return self._l

    @l.setter
    def l(self, x):
        self._l = self.__valid_scalar(x)

    @property
    def length(self):
        """
        :py:class:`float`: Length of the box.
        """
        return self._l

    @length.setter
    def length(self, x):
        self.l = x

    @property
    def w(self):
        """
        :py:class:`float`: Syntactic sugar for width of the box.
        """
        return self._w

    @w.setter
    def w(self, x):
        self._w = self.__valid_scalar(x)

    @property
    def width(self):
        """
        :py:class:`float`: The width of the box.
        """
        return self._w

    @width.setter
    def width(self, x):
        self.w = x

    @property
    def h(self):
        """
        :py:class:`float`: Syntactic sugar for height of the box.
        """
        return self._h

    @h.setter
    def h(self, x):
        self._h = self.__valid_scalar(x)

    @property
    def height(self):
        """
        :py:class:`float`: The height of the box.
        """
        return self._h

    @height.setter
    def height(self, x):
        self.h = x

    def __transform(self, x):
        """
        Rotate and translate the point to world coordinates.
        """
        y = self._c + self._q.rotate(x)
        return y

    @property
    def p1(self):
        """
        :py:class:`float`: Back-left-bottom point.
        """
        p = np.array([-self._l/2, -self._w/2, -self._h/2])
        p = self.__transform(p)
        return p

    @property
    def p2(self):
        """
        :py:class:`float`: Front-left-bottom point.
        """
        p = np.array([self._l/2, -self._w/2, -self._h/2])
        p = self.__transform(p)
        return p

    @property
    def p3(self):
        """
        :py:class:`float`: Front-right-bottom point.
        """
        p = np.array([self._l/2, self._w/2, -self._h/2])
        p = self.__transform(p)
        return p

    @property
    def p4(self):
        """
        :py:class:`float`: Back-right-bottom point.
        """
        p = np.array([-self._l/2, self._w/2, -self._h/2])
        p = self.__transform(p)
        return p

    @property
    def p5(self):
        """
        :py:class:`float`: Back-left-top point.
        """
        p = np.array([-self._l/2, -self._w/2, self._h/2])
        p = self.__transform(p)
        return p

    @property
    def p6(self):
        """
        :py:class:`float`: Front-left-top point.
        """
        p = np.array([self._l/2, -self._w/2, self._h/2])
        p = self.__transform(p)
        return p

    @property
    def p7(self):
        """
        :py:class:`float`: Front-right-top point.
        """
        p = np.array([self._l/2, self._w/2, self._h/2])
        p = self.__transform(p)
        return p

    @property
    def p8(self):
        """
        :py:class:`float`: Back-right-top point.
        """
        p = np.array([-self._l/2, self._w/2, self._h/2])
        p = self.__transform(p)
        return p

    @property
    def p(self):
        """
        Attribute to access ndarray of all corners of box in order.

        Returns:
            :py:class:`ndarray` of float: All corners of the bounding box in order.
        """
        x = np.vstack([self.p1, self.p2, self.p3, self.p4,
                       self.p5, self.p6, self.p7, self.p8])
        return x

    def __repr__(self):
        template = "BBox3D(x={cx}, y={cy}, z={cz}), length={l}, width={w}, height={h}, "\
            "q=({rw}, {rx}, {ry}, {rz}))"
        return template.format(
            cx=self._cx, cy=self._cy, cz=self._cz,
            l=self._l, w=self._w, h=self._h,
            rw=self._q.real,
            rx=self._q.imaginary[0], ry=self._q.imaginary[1], rz=self._q.imaginary[2])

    def copy(self):
        return deepcopy(self)
