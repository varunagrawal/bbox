import numpy as np
from bbox import BBox2D, BBox2DList
import pytest


class TestBBox2DList(object):
    @classmethod
    def setup_class(cls):
        cls.n = 10
        cls.l = [BBox2D(np.random.randint(0, 1024, size=4))
                 for _ in range(cls.n)]
        cls.bbl = BBox2DList(cls.l)

    def test_null(self):
        bbl = BBox2DList([])
        assert bbl.shape == (0, 4)

    def test_len(self):
        assert len(self.bbl) == self.n

    def test_box_shapes(self):
        n = 10
        l = [BBox2D(np.random.randint(0, 1024, size=4)) for _ in range(n)]
        bbl = BBox2DList(l)

        assert bbl.shape == (n, 4)

        lx1 = np.array([b.x1 for b in l])
        lx2 = np.array([b.x2 for b in l])
        ly1 = np.array([b.y1 for b in l])
        ly2 = np.array([b.y2 for b in l])

        assert lx1.shape == bbl.x1.shape
        assert ly1.shape == bbl.y1.shape
        assert lx2.shape == bbl.x2.shape
        assert ly2.shape == bbl.y2.shape

        assert np.array_equal(lx1, bbl.x1)
        assert np.array_equal(lx2, bbl.x2)
        assert np.array_equal(ly1, bbl.y1)
        assert np.array_equal(ly2, bbl.y2)

        assert bbl.x1.shape == (n,)

    def test_equality(self):
        bblist = BBox2DList(self.l)
        assert bblist == self.bbl

    def test_getitem(self):
        assert self.bbl[3] == self.l[3]

    def test_getitem_invalid_key(self):
        with pytest.raises(IndexError):
            self.bbl['random']
        with pytest.raises(IndexError):
            self.bbl[30]

    def test_setitem(self):
        self.bbl[0] = [5, 6, 7, 8]
        self.bbl[1] = BBox2D([1, 2, 3, 4])
        assert np.array_equal(self.bbl[0], BBox2D([5, 6, 7, 8]))
        assert np.array_equal(self.bbl[1], BBox2D([1, 2, 3, 4]))

    def test_x1(self):
        assert np.array_equal(self.bbl.x1, self.bbl.bboxes[:, 0])

    def test_y1(self):
        assert np.array_equal(self.bbl.y1, self.bbl.bboxes[:, 1])

    def test_x2(self):
        assert np.array_equal(self.bbl.x2, self.bbl.bboxes[:, 2])

    def test_y2(self):
        assert np.array_equal(self.bbl.y2, self.bbl.bboxes[:, 3])

    def test_width(self):
        w = self.bbl.bboxes[:, 2] - self.bbl.bboxes[:, 0] + 1
        h = self.bbl.bboxes[:, 3] - self.bbl.bboxes[:, 1] + 1
        assert np.array_equal(self.bbl.w, w)
        assert np.array_equal(self.bbl.width, w)

        assert np.array_equal(self.bbl.h, h)
        assert np.array_equal(self.bbl.height, h)
