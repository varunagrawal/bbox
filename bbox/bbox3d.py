import numpy as np
from pyquaternion import Quaternion


class BBox3D:
    """
    Class for 3D Bounding Boxes
    """

    def __init__(self, x, y, z, length, width, height, rx=0, ry=0, rz=0, rw=1, center=False):
        """
        For now we just take either the center of the 3D bounding box or the top-left-closer corner,
        and the width, height and length, and quaternion values.
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
        self._q = Quaternion(rw, rx, ry, rz)

    @property
    def center(self):
        return np.array([self._cx, self._cy, self._cz, 1])

    @property
    def p1(self):
        p = np.array([self._l/2, -self._w/2, -self._h/2])
        p = self._q.rotate(p)
        p = p + self._c
        return np.append(p, 1)

    @property
    def p2(self):
        p = np.array([self._l/2, self._w/2, -self._h/2])
        p = self._q.rotate(p)
        p = p + self._c
        return np.append(p, 1)

    @property
    def p3(self):
        p = np.array([-self._l/2, self._w/2, -self._h/2])
        p = self._q.rotate(p)
        p = p + self._c
        return np.append(p, 1)

    @property
    def p4(self):
        p = np.array([-self._l/2, -self._w/2, -self._h/2])
        p = self._q.rotate(p)
        p = p + self._c
        return np.append(p, 1)

    @property
    def p5(self):
        p = np.array([self._l/2, -self._w/2, self._h/2])
        p = self._q.rotate(p)
        p = p + self._c
        return np.append(p, 1)

    @property
    def p6(self):
        p = np.array([self._l/2, self._w/2, self._h/2])
        p = self._q.rotate(p)
        p = p + self._c
        return np.append(p, 1)

    @property
    def p7(self):
        p = np.array([-self._l/2, self._w/2, self._h/2])
        p = self._q.rotate(p)
        p = p + self._c
        return np.append(p, 1)

    @property
    def p8(self):
        p = np.array([-self._l/2, -self._w/2, self._h/2])
        p = self._q.rotate(p)
        p = p + self._c
        return np.append(p, 1)

    # def draw_cuboid(img, p):
    #     draw = ImageDraw.Draw(img)
    #     color = tuple(np.random.choice(range(256), size=3))

    #     draw.line([p[0][0], p[0][1], p[1][0], p[1][1]], fill=color, width=2)
    #     draw.line([p[1][0], p[1][1], p[5][0], p[5][1]], fill=color, width=2)
    #     draw.line([p[5][0], p[5][1], p[4][0], p[4][1]], fill=color, width=2)
    #     draw.line([p[4][0], p[4][1], p[0][0], p[0][1]], fill=color, width=2)

    #     draw.line([p[3][0], p[3][1], p[2][0], p[2][1]], fill=color, width=2)
    #     draw.line([p[2][0], p[2][1], p[6][0], p[6][1]], fill=color, width=2)
    #     draw.line([p[6][0], p[6][1], p[7][0], p[7][1]], fill=color, width=2)
    #     draw.line([p[7][0], p[7][1], p[3][0], p[3][1]], fill=color, width=2)

    #     draw.line([p[0][0], p[0][1], p[3][0], p[3][1]], fill=color, width=2)
    #     draw.line([p[1][0], p[1][1], p[2][0], p[2][1]], fill=color, width=2)
    #     draw.line([p[5][0], p[5][1], p[6][0], p[6][1]], fill=color, width=2)
    #     draw.line([p[4][0], p[4][1], p[7][0], p[7][1]], fill=color, width=2)
    #     return img
