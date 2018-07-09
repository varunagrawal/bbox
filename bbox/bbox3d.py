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

    @property
    def q(self):
        """Return the rotation quaternion"""
        return np.hstack((self._q.real, self._q.imaginary))

    @property
    def quaternion(self):
        return self.q

    @property
    def cx(self):
        return self._cx

    @property
    def cy(self):
        return self._cy

    @property
    def cz(self):
        return self._cz

    @property
    def l(self):
        return self._l

    @property
    def length(self):
        return self._l

    @property
    def w(self):
        return self._w

    @property
    def width(self):
        return self._w

    @property
    def h(self):
        return self._h

    @property
    def height(self):
        return self._h

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
