import numpy as np


class BBox3D:
    """
    Class for 3D Bounding Boxes
    """
    def __init__(self, x, y, z, width, height, length, alpha, beta, gamma, center=False):
        """
        For now we just take either the center of the 3D bounding box or the top-left-closer corner,
        and the width, height and length, and euler angles.
        """
        if center:
            self._cx, self._cy, self._cz = x, y, z
        else:
            self._cx = x + width/2
            self._cy = y + height/2
            self._cz = z + length/2

        self._w, self._h, self._l = width, height, length
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def rotate_x(self):
        rot_x = np.array([[1, 0, 0],
                          [0, np.cos(self.alpha), -np.sin(self.alpha)],
                          [0, np.sin(self.alpha), np.cos(self.alpha)]])
        u = np.array([[self._cx, self._cy, self._cz]]).T
        v = rot_x @ u
        self._cx, self._cy, self._cz = v 

    @property
    def x1(self):
        return self._cx - self._w/2

    @property
    def x2(self):
        return self._cx + self._w/2

    @property
    def x3(self):
        return self._cx - self._w/2

    @property
    def x4(self):
        return self._cx + self._w/2

    @property
    def x5(self):
        return self._cx - self._w/2

    @property
    def x6(self):
        return self._cx + self._w/2

    @property
    def x7(self):
        return self._cx - self._w/2

    @property
    def x8(self):
        return self._cx + self._w/2

    @property
    def y1(self):
        return self._cy - self._h/2
    
    @property
    def y2(self):        
        return self._cy - self._h/2
    
    @property
    def y3(self):
        return self._cy + self._h/2

    @property
    def y4(self):
        return self._cy + self._h/2

    @property
    def y5(self):
        return self._cy - self._h/2

    @property
    def y6(self):
        return self._cy - self._h/2

    @property
    def y7(self):
        return self._cy + self._h/2

    @property
    def y8(self):
        return self._cy + self._h/2

    @property
    def z1(self):
        return self._cz - self._l/2

    @property
    def z2(self):
        return self._cz - self._l/2

    @property
    def z3(self):
        return self._cz - self._l/2

    @property
    def z4(self):
        return self._cz - self._l/2

    @property
    def z5(self):
        return  self._cz + self._l/2

    @property
    def z6(self):
        return self._cz + self._l/2

    @property
    def z7(self):
        return self._cz + self._l/2

    @property
    def z8(self):
        return self._cz + self._l/2


