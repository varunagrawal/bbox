import pytest
import numpy as np
from bbox import BBox2D, BBox2DList
from bbox.box_modes import XYXY, XYWH


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

    def test_init(self):
        bbl = BBox2DList(self.bbl)
        assert np.array_equal(bbl.numpy(), self.bbl.numpy())

    def test_init_invalid(self):
        with pytest.raises(TypeError):
            BBox2DList("1, 2, 3, 4")

    def test_init_invalid_element_type(self):
        with pytest.raises(TypeError):
            BBox2DList(["1, 2, 3, 4", [1, 2, 3, 4]])

    def test_init_empty_ndarray(self):
        bbl = BBox2DList(np.empty((0, 4)))
        assert bbl.bboxes.shape == (0, 4)

    def test_init_vector(self):
        bbl = BBox2DList(np.asarray([0, 1, 2, 4]))
        assert bbl.bboxes.shape == (1, 4)

    def test_init_invalid_dims(self):
        with pytest.raises(ValueError):
            BBox2DList(np.random.rand(10, 3))
        with pytest.raises(ValueError):
            BBox2DList(np.random.rand(10, 5))
        with pytest.raises(ValueError):
            BBox2DList(np.random.rand(10, 1, 4))

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

    def test_inequality(self):
        bbl = BBox2DList([BBox2D(np.random.randint(0, 1024, size=4))
                          for _ in range(self.n)])
        assert bbl != self.bbl

    def test_equality_invalid(self):
        bblist = BBox2DList(self.l)
        assert bblist != repr(self.bbl)

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

    def test_x1_getter(self):
        assert np.array_equal(self.bbl.x1, self.bbl.bboxes[:, 0])

    def test_x1_setter(self):
        bbl = self.bbl.copy()
        bbl.x1 = np.zeros(bbl.shape[0])
        assert np.array_equal(bbl.x1, np.zeros(bbl.shape[0]))

    def test_y1_getter(self):
        assert np.array_equal(self.bbl.y1, self.bbl.bboxes[:, 1])

    def test_y1_setter(self):
        bbl = self.bbl.copy()
        bbl.y1 = np.zeros(bbl.shape[0])
        assert np.array_equal(bbl.y1, np.zeros(bbl.shape[0]))

    def test_x2_getter(self):
        assert np.array_equal(self.bbl.x2, self.bbl.bboxes[:, 2])

    def test_x2_setter(self):
        bbl = self.bbl.copy()
        bbl.x2 = np.zeros(bbl.shape[0])
        assert np.array_equal(bbl.x2, np.zeros(bbl.shape[0]))

    def test_y2_getter(self):
        assert np.array_equal(self.bbl.y2, self.bbl.bboxes[:, 3])

    def test_y2_setter(self):
        bbl = self.bbl.copy()
        bbl.y2 = np.zeros(bbl.shape[0])
        assert np.array_equal(bbl.y2, np.zeros(bbl.shape[0]))

    def test_invalid_setter(self):
        """
        One test is sufficient since all setters use the same verification function
        """
        bbl = self.bbl.copy()
        with pytest.raises(TypeError):
            bbl.x1 = "0," * self.bbl.shape[0]
        with pytest.raises(ValueError):
            bbl.x1 = np.zeros((5, 4))
        with pytest.raises(ValueError):
            bbl.x1 = np.zeros(5)

    def test_width_getter(self):
        w = self.bbl.bboxes[:, 2] - self.bbl.bboxes[:, 0]
        assert np.array_equal(self.bbl.w, w)
        assert np.array_equal(self.bbl.width, w)

    def test_width_setter(self):
        bbl = self.bbl.copy()
        w = np.ones(bbl.shape[0])
        bbl.w = w
        assert np.array_equal(bbl.w, w)
        assert np.array_equal(bbl.width, w)

    def test_height_getter(self):
        h = self.bbl.bboxes[:, 3] - self.bbl.bboxes[:, 1]
        assert np.array_equal(self.bbl.h, h)
        assert np.array_equal(self.bbl.height, h)

    def test_height_setter(self):
        bbl = self.bbl.copy()
        h = np.ones(bbl.shape[0])
        bbl.h = h
        assert np.array_equal(bbl.h, h)
        assert np.array_equal(bbl.height, h)

    def test_mul(self):
        bbl = BBox2DList(np.ones((7, 4)))
        bbl_scaled = bbl * 11
        assert np.all(bbl_scaled.bboxes[:, 0:2] == 11)
        assert np.all(bbl_scaled.bboxes[:, 2:4] == 22)
        bbl_scaled = 11 * bbl
        assert np.all(bbl_scaled.bboxes[:, 0:2] == 11)
        assert np.all(bbl_scaled.bboxes[:, 2:4] == 22)

    def test_invalid_mul(self):
        bbl = BBox2DList(np.ones((7, 4)))
        with pytest.raises(ValueError):
            bbl * "11"

    def test_append_list(self):
        x = [3, 7, 10, 44]
        bbl = self.bbl.append(x, mode=XYXY)
        assert np.array_equal(bbl.bboxes[-1], x)

    def test_append_numpy(self):
        x = np.asarray([3, 7, 10, 16])
        bbl = self.bbl.append(x, mode=XYXY)
        assert np.array_equal(bbl.bboxes[-1], x)

    def test_append_bbox(self):
        x = BBox2D([3, 7, 10, 16], mode=XYXY)
        bbl = self.bbl.append(x)
        assert np.array_equal(bbl.bboxes[-1], x.numpy(mode=XYXY))

    def test_append_bboxlist(self):
        x = BBox2DList([[3, 7, 10, 16]], mode=XYXY)
        bbl = self.bbl.append(x)
        assert np.array_equal(bbl.bboxes,
                              np.vstack((self.bbl.bboxes,
                                         [3, 7, 10, 16])))

    def test_append_invalid(self):
        x = "3, 7, 10, 16"
        with pytest.raises(TypeError):
            bbl = self.bbl.append(x, mode=XYXY)

    def test_append_invalid_list(self):
        x = ["abc", "7", 10, 16]
        with pytest.raises(ValueError):
            bbl = self.bbl.append(x, mode=XYXY)

    def test_append_invalid_range(self):
        x = range(4)
        with pytest.raises(TypeError):
            self.bbl.append(x)

    def test_append_invalid_dimensions(self):
        with pytest.raises(ValueError):
            self.bbl.append((1, 2, 3))

        with pytest.raises(ValueError):
            self.bbl.append((1, 2, 3, 4, 5))

        with pytest.raises(ValueError):
            self.bbl.append([[1, 2, 3, 4, 5]])

    def test_insert_list(self):
        x = [3, 7, 10, 16]
        bbl = self.bbl.insert(x, 0, mode=XYXY)
        assert np.array_equal(bbl.bboxes[0], x)

    def test_insert_numpy(self):
        x = np.asarray([3, 7, 10, 16])
        bbl = self.bbl.insert(x, 0, mode=XYXY)
        assert np.array_equal(bbl.bboxes[0], x)

    def test_insert_bbox(self):
        x = BBox2D([3, 7, 10, 16], mode=XYXY)
        bbl = self.bbl.insert(x, 0)
        assert np.array_equal(bbl.bboxes[0], x.numpy(mode=XYXY))

    def test_insert_invalid_datatype(self):
        x = range(4)
        with pytest.raises(TypeError):
            self.bbl.insert(x, 0)
        with pytest.raises(TypeError):
            self.bbl.insert("abcd", 0)

    def test_insert_invalid_type(self):
        with pytest.raises(ValueError):
            self.bbl.insert(["a", "b", "c", "d"], 0)

    def test_insert_invalid_dimensions(self):
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

    def test_str(self):
        bbl = BBox2DList([[0, 0, 1, 1], [5, 5, 5, 5]])
        assert str(bbl) == "[[0. 0. 1. 1.]\n [5. 5. 5. 5.]]"

    def test_repr(self):
        bbl = BBox2DList([[0, 0, 1, 1], [5, 5, 5, 5]])
        assert repr(
            bbl) == "array([[0., 0., 1., 1.],\n       [5., 5., 5., 5.]])"
