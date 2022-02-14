import pytest
import numpy as np
from bbox import BBox2D, BBox2DList
from bbox.box_modes import XYXY, XYWH


class TestBBox2D(object):
    def attributes_test(self, bbox, x1, y1, x2, y2, w, h):
        """
        Convenience method to test all the bounding box attributes of importance.
        """
        assert bbox.x1 == x1
        assert bbox.x2 == x2
        assert bbox.y1 == y1
        assert bbox.y2 == y2
        assert bbox.w == w
        assert bbox.h == h
        assert bbox.width == w
        assert bbox.height == h

    def test_constructor_empty_list(self):
        with pytest.raises(ValueError):
            BBox2D([])

    def test_constructor_1_item(self):
        with pytest.raises(ValueError):
            BBox2D([1])

    def test_constructor_2_items(self):
        with pytest.raises(ValueError):
            BBox2D([1, 2])

    def test_constructor_3_items(self):
        with pytest.raises(ValueError):
            BBox2D([1, 2, 4])

    def test_constructor_5_items(self):
        with pytest.raises(ValueError):
            BBox2D([1, 2, 3, 4, 5])
        with pytest.raises(ValueError):
            BBox2D(np.array([1, 2, 3, 4, 5]))

    def test_constructor_invalid_type(self):
        with pytest.raises(TypeError):
            BBox2D("1, 2, 3, 4")

    def test_copy_constructor(self):
        bbox1 = BBox2D([10, 11, 510, 511])
        bbox2 = BBox2D(bbox1)
        self.attributes_test(bbox2, 10, 11, 519, 521, 510, 511)

    def test_basic_box(self):
        bbox1 = BBox2D([0, 0, 500, 500])
        self.attributes_test(bbox1, 0, 0, 499, 499, 500, 500)

    def test_nonbasic_box(self):
        bbox2 = BBox2D([24, 48, 64, 96])
        self.attributes_test(bbox2, 24, 48, 87, 143, 64, 96)

    def test_equality(self):
        b1 = BBox2D([1, 2, 3, 4])
        b2 = BBox2D([1, 2, 3, 4])
        assert b1 == b2
        b3 = BBox2D([1, 2, 3, 5])
        assert b1 != b3

    def test_non_equality(self):
        b = BBox2D([1, 2, 3, 4])
        assert (b == [1, 2, 3, 4]) == False

    def test_x1(self):
        bbox = BBox2D([24, 48, 64, 96])
        bbox.x1 = 25
        assert bbox.x1 == 25 and bbox.w == 63

    def test_invalid_x1(self):
        bbox = BBox2D([24, 48, 64, 96])
        with pytest.raises(ValueError):
            bbox.x1 = bbox.x2 + 1

    def test_x2(self):
        bbox = BBox2D([24, 48, 64, 96])
        bbox.x2 = 89
        assert bbox.x2 == 89 and bbox.w == 66

    def test_invalid_x2(self):
        bbox = BBox2D([24, 48, 64, 96])
        with pytest.raises(ValueError):
            bbox.x2 = bbox.x1 - 1

    def test_y1(self):
        bbox = BBox2D([24, 48, 64, 96])
        bbox.y1 = 51
        assert bbox.y1 == 51 and bbox.h == 93

    def test_invalid_y1(self):
        bbox = BBox2D([24, 48, 64, 96])
        with pytest.raises(ValueError):
            bbox.y1 = bbox.y2 + 1

    def test_y2(self):
        bbox = BBox2D([24, 48, 64, 30])
        bbox.y2 = 80
        assert bbox.y2 == 80 and bbox.h == 33

    def test_invalid_y2(self):
        bbox = BBox2D([24, 48, 64, 96])
        with pytest.raises(ValueError):
            bbox.y2 = bbox.y1 - 1

    def test_w(self):
        bbox = BBox2D([24, 48, 64, 96])
        bbox.width = 70
        assert bbox.width == 70 and bbox.x1 == 24 and bbox.x2 == 93
        bbox.w = 70
        assert bbox.width == 70 and bbox.w == 70

        with pytest.raises(ValueError):
            bbox.w = -1
        with pytest.raises(ValueError):
            bbox.width = -1

    def test_h(self):
        bbox = BBox2D([24, 48, 64, 96])
        bbox.height = 93
        assert bbox.height == 93 and bbox.y1 == 48 and bbox.y2 == 140
        bbox.h = 93
        assert bbox.height == 93 and bbox.h == 93

        with pytest.raises(ValueError):
            bbox.h = -1
        with pytest.raises(ValueError):
            bbox.height = -1

    def test_tolist(self):
        bbox = BBox2D([24, 48, 64, 96])
        bbox_list = bbox.tolist(mode=XYWH)
        for x, y in zip(bbox_list, [24, 48, 64, 96]):
            assert x == y
        bbox_list_2 = bbox.tolist(mode=XYXY)
        for x, y in zip(bbox_list_2, [24, 48, 87, 143]):
            assert x == y

    def test_numpy(self):
        x = np.array([24, 48, 64, 96])
        bbox = BBox2D(x)
        bb_np = bbox.numpy(mode=XYWH)
        assert np.array_equal(bb_np, x)

        bbox = BBox2D(x, mode=XYXY)
        bb_np_2 = bbox.numpy(mode=XYXY)
        assert np.array_equal(bb_np_2, x)

    def test_str(self):
        bbox = BBox2D([24, 48, 64, 96])
        s = repr(bbox)
        assert "BBox2D([24.0, 48.0, 64.0, 96.0])" == s

    def test_aspect_ratio(self):
        bbox = BBox2D([0, 0, 16, 16])

        new_bbox = bbox.aspect_ratio(1)
        assert new_bbox.w == 16 and new_bbox.h == 16

        new_bbox = bbox.aspect_ratio(0.5)
        assert new_bbox.w == 11 and new_bbox.h == 22

        new_bbox = bbox.aspect_ratio(2)
        assert new_bbox.w == 23 and new_bbox.h == 12

    def test_copy(self):
        b1 = BBox2D([24, 48, 64, 96])
        b2 = b1.copy()

        assert b1 == b2

    def test_mul(self):
        bbox = BBox2D([24, 48, 64, 96], mode=XYXY)
        scaled_bbox = bbox * 2
        assert np.array_equal(scaled_bbox.numpy(mode=XYXY),
                              np.array([48, 96, 128, 192],
                                       dtype=float))
        scaled_bbox_left = 2 * bbox
        assert np.array_equal(scaled_bbox_left.numpy(mode=XYXY),
                              np.array([48, 96, 128, 192],
                                       dtype=float))

    def test_invalid_mul(self):
        bbox = BBox2D([24, 48, 64, 96])
        with pytest.raises(ValueError):
            bbox * (1, 2)

    def test_contains_point(self):
        bbox = BBox2D([24, 48, 64, 96])
        with pytest.raises(ValueError):
            bbox.contains(1)

        with pytest.raises(ValueError):
            bbox.contains([1, 2, 3])

        # test tuple
        assert bbox.contains((35, 57))
        # test list
        assert bbox.contains([35, 57])
        # test numpy array
        assert bbox.contains(np.array([35, 57]))

        assert not bbox.contains([22, 57])
        assert not bbox.contains([25, 47])
        assert not bbox.contains([90, 57])
        assert not bbox.contains([25, 144])
