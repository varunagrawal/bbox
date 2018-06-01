import numpy as np
from bbox import BoundingBox, BoundingBoxList


class TestBoundingBoxList(object):
    def test_null(self):
        bbl = BoundingBoxList.from_bbox_list([])
        print(bbl.shape)

    def test_box_shapes(self):
        l = [BoundingBox(*np.random.randint(0, 1024, size=4)) for _ in range(10)]
        bbl = BoundingBoxList.from_bbox_list(l)

        assert len(bbl) == 10
        assert bbl.shape == (10, 4)

        lx1 = np.array([b.x1 for b in l])
        lx2 = np.array([b.x2 for b in l])
        ly1 = np.array([b.y1 for b in l])
        ly2 = np.array([b.y2 for b in l])

        assert np.array_equal(lx1, bbl.x1)
        assert np.array_equal(lx2, bbl.x2)
        assert np.array_equal(ly1, bbl.y1)
        assert np.array_equal(ly2, bbl.y2)