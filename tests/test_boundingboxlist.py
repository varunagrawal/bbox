import numpy as np
from bbox import BBox2D, BBox2DList


class TestBBox2DList(object):
    def test_null(self):
        bbl = BBox2DList([])
        assert bbl.shape == (0, 4)

    def test_box_shapes(self):
        l = [BBox2D(np.random.randint(0, 1024, size=4)) for _ in range(10)]
        bbl = BBox2DList(l)

        assert len(bbl) == 10
        assert bbl.shape == (10, 4)

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

        assert bbl.x1.shape == (10,)