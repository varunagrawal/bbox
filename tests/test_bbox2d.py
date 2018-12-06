import pytest
import numpy as np
from bbox import BBox2D, BBox2DList


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

    def test_x1(self):
        bbox = BBox2D([24, 48, 64, 96])
        bbox.x1 = 25
        assert bbox.x1 == 25
        assert bbox.w == 63

    def test_x2(self):
        bbox = BBox2D([24, 48, 64, 96])
        bbox.x2 = 89
        assert bbox.x2 == 89
        assert bbox.w == 66

    def test_y1(self):
        bbox = BBox2D([24, 48, 64, 96])
        bbox.y1 = 51
        assert bbox.y1 == 51
        assert bbox.h == 93

    def test_y2(self):
        bbox = BBox2D([24, 48, 64, 30])
        bbox.y2 = 80
        assert bbox.y2 == 80
        assert bbox.h == 33

    def test_w(self):
        bbox = BBox2D([24, 48, 64, 96])
        bbox.w = 70
        assert bbox.w == 70
        assert bbox.x1 == 24
        assert bbox.x2 == 93

    def test_h(self):
        bbox = BBox2D([24, 48, 64, 96])
        bbox.h = 93
        assert bbox.h == 93
        assert bbox.y2 == 140

    def test_tolist(self):
        bbox = BBox2D([24, 48, 64, 96])
        bbox_list = bbox.tolist(two_point=False)
        for x, y in zip(bbox_list, [24, 48, 64, 96]):
            assert x == y
        bbox_list_2 = bbox.tolist(two_point=True)
        for x, y in zip(bbox_list_2, [24, 48, 87, 143]):
            assert x == y

    def test_numpy(self):
        x = np.array([24, 48, 64, 96])
        bbox = BBox2D(x)
        bb_np = bbox.numpy(two_point=False)
        assert np.array_equal(bb_np, x)

        bbox = BBox2D(x, two_point=True)
        bb_np_2 = bbox.numpy(two_point=True)
        assert np.array_equal(bb_np_2, x)

    def test_str(self):
        bbox = BBox2D([24, 48, 64, 96])
        s = repr(bbox)
        assert "BBox2D(x=24.0, y=48.0, w=64.0, h=96.0)" == s

    def test_aspect_ratio(self):
        bbox = BBox2D([0, 0, 16, 16])

        new_bbox = bbox.aspect_ratio(1)
        assert new_bbox.w == 16 and new_bbox.h == 16

        new_bbox = bbox.aspect_ratio(0.5)
        assert new_bbox.w == 23 and new_bbox.h == 12

        new_bbox = bbox.aspect_ratio(2)
        assert new_bbox.w == 11 and new_bbox.h == 22
