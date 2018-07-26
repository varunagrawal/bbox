from bbox import BBox3D
from bbox.utils import draw_cuboid
from bbox.metrics import jaccard_index_3d

import numpy as np
from PIL import Image, ImageDraw
import pytest


class TestBBox3d:
    @classmethod
    def setup_class(cls):
        # sample cuboid
        cuboid = {
            'center': {
                'x': -49.19743041908411,
                'y': 12.38666074615689,
                'z': 0.782056864653507
            },
            'dimensions': {
                'length': 5.340892485711914,
                'width': 2.457703972075464,
                'height': 1.9422248281533563
            },
            'rotation': {
                'w': 0.9997472337219893,
                'x': 0.0,
                'y': 0.0,
                'z': 0.022482630300529462
            }
        }
        center = cuboid['center']
        dim = cuboid['dimensions']
        rotation = cuboid['rotation']

        cls.box = BBox3D(center['x'], center['y'], center['z'],
                         length=dim['length'], width=dim['width'], height=dim['height'],
                         rw=rotation['w'], rx=rotation['x'], ry=rotation['y'], rz=rotation['z'])
        cls.cuboid = cuboid

    def test_points(self):
        points = np.array([[-51.80993533, 11.03900409,  -0.18905555],
                           [-46.47444215, 11.27909801,  -0.18905555],
                           [-46.58492551, 13.7343174,   -0.18905555],
                           [-51.92041869, 13.49422348,  -0.18905555],
                           [-51.80993533, 11.03900409,  1.75316928],
                           [-46.47444215, 11.27909801,  1.75316928],
                           [-46.58492551, 13.7343174,   1.75316928],
                           [-51.92041869, 13.49422348,  1.75316928]])

        assert np.allclose(self.box.p1, points[0, :])
        assert np.allclose(self.box.p2, points[1, :])
        assert np.allclose(self.box.p3, points[2, :])
        assert np.allclose(self.box.p4, points[3, :])
        assert np.allclose(self.box.p5, points[4, :])
        assert np.allclose(self.box.p6, points[5, :])
        assert np.allclose(self.box.p7, points[6, :])
        assert np.allclose(self.box.p8, points[7, :])

        assert np.allclose(self.box.p, points)

    def test_center(self):
        center = np.array(
            [-49.19743041908411, 12.38666074615689, 0.782056864653507])
        assert np.array_equal(self.box.center, center)

    def test_center_points(self):
        assert self.box.cx == self.cuboid['center']['x']
        assert self.box.cy == self.cuboid['center']['y']
        assert self.box.cz == self.cuboid['center']['z']

    def test_center_init(self):
        box = BBox3D(*[self.box.cx, self.box.cy, self.box.cz])
        assert np.array_equal(box.center, self.box.center)

    def test_dimensions(self):
        assert self.box.l == self.cuboid['dimensions']['length']
        assert self.box.length == self.cuboid['dimensions']['length']
        assert self.box.w == self.cuboid['dimensions']['width']
        assert self.box.width == self.cuboid['dimensions']['width']
        assert self.box.h == self.cuboid['dimensions']['height']
        assert self.box.height == self.cuboid['dimensions']['height']

    def test_quaternion(self):
        q = np.array([0.9997472337219893, 0.0, 0.0, 0.022482630300529462])
        assert np.array_equal(self.box.q, q)
        # alternative attribute for the quaternion
        assert np.array_equal(self.box.quaternion, q)

        box = BBox3D(self.box.cx, self.box.cy, self.box.cz, q=self.box.q)
        assert np.array_equal(self.box.q, box.q)

    def test_euler_angles(self):
        box = BBox3D(3.163, z=2.468, y=34.677, height=1.529, width=1.587, length=3.948,
                     euler_angles=[0, 0, -1.59])
        q = np.array([0.7002847660410397, -0.0, -0.0, -0.7138636049350369])
        assert np.array_equal(box.q, q)

    def test_projection(self):
        K = np.array([[1406.3359, 0.0, 966.366034, 0.0],
                      [0.0, 1408.94297, 607.479746, 0.0],
                      [0.0, 0.0, 1.0, 0.0]])

        R = np.array([[0.50478576,  0.86323317, -0.00445338],
                      [-0.00422247, -0.00268975, -0.99998747],
                      [-0.86323433,  0.50479824,  0.00228723]])

        t = np.array([[-0.75116634], [1.35776453], [0.87137971]])

        u = np.empty((8, 2))
        u[0] = self.project(self.box.p1, K, R, t)
        u[1] = self.project(self.box.p2, K, R, t)
        u[2] = self.project(self.box.p3, K, R, t)
        u[3] = self.project(self.box.p4, K, R, t)
        u[4] = self.project(self.box.p5, K, R, t)
        u[5] = self.project(self.box.p6, K, R, t)
        u[6] = self.project(self.box.p7, K, R, t)
        u[7] = self.project(self.box.p8, K, R, t)

        image_points = np.array([[488.84269983, 655.2790429],
                                 [530.3490365,  659.17142666],
                                 [602.90920984, 657.55445187],
                                 [556.26022377, 653.89914093],
                                 [488.64644505, 601.79933826],
                                 [530.12998103, 600.55433066],
                                 [602.68953074, 600.56675254],
                                 [556.06325411, 601.7790501]])

        for i in range(8):
            assert np.allclose(u[i], image_points[i])

    def test_setters(self):
        x, y, z, h, w, l = np.random.rand(6)
        self.box.cx = x
        self.box.cy = y
        self.box.cz = z
        self.box.h = h
        self.box.w = w
        self.box.l = l
        assert self.box.cx == x
        assert self.box.cy == y
        assert self.box.cz == z
        assert self.box.h == h
        assert self.box.w == w
        assert self.box.l == l

        q = np.random.rand(4)
        self.box.q = q
        assert np.array_equal(self.box.q, q)
        q = np.random.rand(4)
        self.box.quaternion = q
        assert np.array_equal(self.box.quaternion, q)

        center = np.random.rand(3)
        self.box.center = center
        assert np.array_equal(self.box.center, center)

    def test_bad_setters(self):
        inputs = [[1, 2], np.zeros((3)), np.zeros(
            (3, 1)), np.zeros((1, 3)), "center"]
        for x in inputs:
            with pytest.raises((ValueError, TypeError)):
                self.box.cx = x
            with pytest.raises((ValueError, TypeError)):
                self.box.cy = x
            with pytest.raises((ValueError, TypeError)):
                self.box.cz = x
            with pytest.raises((ValueError, TypeError)):
                self.box.h = x
            with pytest.raises((ValueError, TypeError)):
                self.box.w = x
            with pytest.raises((ValueError, TypeError)):
                self.box.l = x
            with pytest.raises((ValueError, TypeError)):
                self.box.height = x
            with pytest.raises((ValueError, TypeError)):
                self.box.width = x
            with pytest.raises((ValueError, TypeError)):
                self.box.length = x

        quaternion_inputs = [1, [1], [1, 2, 3], [1, 2, 3, 4, 5], "quaternion"]
        for q in quaternion_inputs:
            with pytest.raises((ValueError, TypeError)):
                self.box.q = q

    def test_non_overlapping_boxes(self):
        a = BBox3D(x=-2.553668269106177, y=-63.56305079381365, z=1.988316894113887,
                   length=4.7, width=1.8420955618567376, height=1.4,
                   q=(-0.7123296970493456, 0.0, 0.0, 0.7018449990571904))

        b = BBox3D(x=-60.00052106600015, y=-4.111285291215302, z=0.7497459084120979,
                   length=4.7, width=1.8, height=1.819601518010064,
                   q=(0.999845654958524, 0.0, 0.0, 0.017568900379933073))
        # print(jaccard_index_3d(a, b))

    @pytest.mark.skip(reason="This is just for visualization. We already test the values beforehand.")
    def test_render(self):
        K = np.array([[1406.3359, 0.0, 966.366034, 0.0],
                      [0.0, 1408.94297, 607.479746, 0.0],
                      [0.0, 0.0, 1.0, 0.0]])

        R = np.array([[0.50478576,  0.86323317, -0.00445338],
                      [-0.00422247, -0.00268975, -0.99998747],
                      [-0.86323433,  0.50479824,  0.00228723]])

        t = np.array([[-0.75116634], [1.35776453], [0.87137971]])

        u = np.empty((8, 2))
        u[0] = self.project(self.box.p1, K, R, t)
        u[1] = self.project(self.box.p2, K, R, t)
        u[2] = self.project(self.box.p3, K, R, t)
        u[3] = self.project(self.box.p4, K, R, t)
        u[4] = self.project(self.box.p5, K, R, t)
        u[5] = self.project(self.box.p6, K, R, t)
        u[6] = self.project(self.box.p7, K, R, t)
        u[7] = self.project(self.box.p8, K, R, t)

        img = Image.open("image_raw_ring_rear_left_315968023469083760.png")

        dist_coeff = [-0.17120984449230167,
                      0.1256910189977147,
                      -0.029726711792577232]

        for i in range(u.shape[0]):
            u[i] = self.distortion_correction(u[i], dist_coeff, img.size)

        img = draw_cuboid(img, u)
        img.show()

    @staticmethod
    def project(p, K, R, t):
        p = np.hstack((p, 1))
        E = np.eye(4)
        E[0: 3, 0: 3] = R
        E[0: 3, 3: 4] = t

        u = K @ E @ p
        # print("projection", u)
        return u[0: 2] / u[2]

    @staticmethod
    def distortion_correction(u, dist_coeff, img_size):
        w, h = img_size
        # normalize the image coords
        x = 2*u[0]/w - 1
        y = 2*u[1]/h - 1
        r = np.linalg.norm(np.array(x, y))

        r2 = r**2

        distortion = 0
        for i in range(len(dist_coeff)):
            distortion += ((r2**(i+1)) * dist_coeff[i])

        # correct for distortion
        v = u + (u - np.array([w, h])/2)*distortion
        return v
