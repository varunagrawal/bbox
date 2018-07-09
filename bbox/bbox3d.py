import numpy as np
from pyquaternion import Quaternion


class BBox3D:
    """
    Class for 3D Bounding Boxes
    """

    def __init__(self, x, y, z, c=None,
                 length=1, width=1, height=1,
                 rw=1, rx=0, ry=0, rz=0,
                 euler_angles=None, center=True):
        """
        For now we just take either the center of the 3D bounding box or the top-left-closer corner, and the width, height and length, and quaternion values.
        """
        if center:
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

        else:
            self._q = Quaternion(rw, rx, ry, rz)

    @property
    def center(self):
        return self._c

    @center.setter
    def center(self, c):
        if len(c) != 3:
            raise ValueError("Center coordinates should be a vector of size 3")
        self._c = c

    @property
    def cx(self):
        return self._cx

    def _is_valid_scalar(self, x):
        if not np.isscalar(x):
            raise ValueError("Value should be a scalar")
        else:  # x is a scalar so we check for numeric type
            if not isinstance(x, (np.float, np.int)):
                raise TypeError("Value needs to be either a float or an int")
        return x

    @cx.setter
    def cx(self, x):
        self._cx = self._is_valid_scalar(x)

    @property
    def cy(self):
        return self._cy

    @cy.setter
    def cy(self, x):
        self._cy = self._is_valid_scalar(x)

    @property
    def cz(self):
        return self._cz

    @cz.setter
    def cz(self, x):
        self._cz = self._is_valid_scalar(x)

    @property
    def q(self):
        """Return the rotation quaternion"""
        return np.hstack((self._q.real, self._q.imaginary))

    @q.setter
    def q(self, q):
        if not isinstance(q, (list, tuple, np.ndarray, Quaternion)):
            raise TypeError("Value shoud be either list, numpy array or Quaterion")
        if isinstance(q, (list, tuple, np.ndarray)) and len(q) != 4:
            raise ValueError("Quaternion input should be a vector of size 4")

        self._q = Quaternion(q)

    @property
    def quaternion(self):
        return self.q

    @quaternion.setter
    def quaternion(self, q):
        self.q = q

    @property
    def l(self):
        return self._l

    @l.setter
    def l(self, x):
        self._l = self._is_valid_scalar(x)

    @property
    def length(self):
        return self._l

    @length.setter
    def length(self, x):
        self.l = x

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, x):
        self._w = self._is_valid_scalar(x)

    @property
    def width(self):
        return self._w

    @width.setter
    def width(self, x):
        self.w = x

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, x):
        self._h = self._is_valid_scalar(x)

    @property
    def height(self):
        return self._h

    @height.setter
    def height(self, x):
        self.h = x

    def _transform(self, x):
        """Rotate and translate the point to world coordinates"""
        y = self._c + self._q.rotate(x)
        return y

    @property
    def p1(self):
        p = np.array([-self._l/2, -self._w/2, -self._h/2])
        p = self._transform(p)
        return p

    @property
    def p2(self):
        p = np.array([self._l/2, -self._w/2, -self._h/2])
        p = self._transform(p)
        return p

    @property
    def p3(self):
        p = np.array([self._l/2, self._w/2, -self._h/2])
        p = self._transform(p)
        return p

    @property
    def p4(self):
        p = np.array([-self._l/2, self._w/2, -self._h/2])
        p = self._transform(p)
        return p

    @property
    def p5(self):
        p = np.array([-self._l/2, -self._w/2, self._h/2])
        p = self._transform(p)
        return p

    @property
    def p6(self):
        p = np.array([self._l/2, -self._w/2, self._h/2])
        p = self._transform(p)
        return p

    @property
    def p7(self):
        p = np.array([self._l/2, self._w/2, self._h/2])
        p = self._transform(p)
        return p

    @property
    def p8(self):
        p = np.array([-self._l/2, self._w/2, self._h/2])
        p = self._transform(p)
        return p

    @property
    def p(self):
        x = np.vstack([self.p1, self.p2, self.p3, self.p4,
                       self.p5, self.p6, self.p7, self.p8])
        return x

    def __repr__(self):
        return "BBox3D(c=({cx},{cy},{cz}), {l}x{w}x{h}, q=[{rw}, {rx}, {ry}, {rz}])".format(
            cx=self._cx, cy=self._cy, cz=self._cz,
            l=self._l, w=self._w, h=self._h,
            rw=self._q.real, rx=self._q.imaginary[0], ry=self._q.imaginary[1], rz=self._q.imaginary[2])
