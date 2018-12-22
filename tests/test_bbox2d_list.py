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

    def test_mul(self):
        bbl = BBox2DList(np.ones((7, 4)))
        bbl_scaled = bbl * 11
        assert np.all(bbl_scaled.bboxes == 11)

    def test_append_list(self):
        x = [3, 7, 10, 16]
        bbl = self.bbl.append(x, two_point=True)
        assert np.array_equal(bbl.bboxes[-1], x)

    def test_append_numpy(self):
        x = np.asarray([3, 7, 10, 16])
        bbl = self.bbl.append(x, two_point=True)
        assert np.array_equal(bbl.bboxes[-1], x)

    def test_append_bbox(self):
        x = BBox2D([3, 7, 10, 16], two_point=True)
        bbl = self.bbl.append(x)
        assert np.array_equal(bbl.bboxes[-1], x.numpy(two_point=True))

    def test_append_invalid_type(self):
        x = range(4)
        with pytest.raises(TypeError):
            self.bbl.append(x)

    def test_append_invalid_value(self):
        with pytest.raises(ValueError):
            self.bbl.append((1, 2, 3))

        with pytest.raises(ValueError):
            self.bbl.append((1, 2, 3, 4, 5))

    def test_insert_list(self):
        x = [3, 7, 10, 16]
        bbl = self.bbl.insert(x, 0, two_point=True)
        assert np.array_equal(bbl.bboxes[0], x)

    def test_insert_numpy(self):
        x = np.asarray([3, 7, 10, 16])
        bbl = self.bbl.insert(x, 0, two_point=True)
        assert np.array_equal(bbl.bboxes[0], x)

    def test_insert_bbox(self):
        x = BBox2D([3, 7, 10, 16], two_point=True)
        bbl = self.bbl.insert(x, 0)
        assert np.array_equal(bbl.bboxes[0], x.numpy(two_point=True))

    def test_insert_invalid_type(self):
        x = range(4)
        with pytest.raises(TypeError):
            self.bbl.insert(x, 0)

    def test_insert_invalid_value(self):
        with pytest.raises(ValueError):
            self.bbl.insert((1, 2, 3), 0)

        with pytest.raises(ValueError):
            self.bbl.insert((1, 2, 3, 4, 5), 0)

    def test_delete(self):
        idx = 5
        bbl = self.bbl.delete(idx)
        assert bbl.shape[0] == self.bbl.shape[0]-1
        assert self.bbl[idx] not in bbl

    def test_delete_negative(self):
        idx = -3
        bbl = self.bbl.delete(idx)
        assert bbl.shape[0] == self.bbl.shape[0]-1
        assert self.bbl[idx] not in bbl

    def test_delete_invalid(self):
        # idx is one greater than the max allowed index
        idx = self.bbl.shape[0]
        with pytest.raises(IndexError):
            bbl = self.bbl.delete(idx)
