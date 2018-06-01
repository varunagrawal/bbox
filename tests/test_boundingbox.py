from bbox import BoundingBox, BoundingBoxList


class TestBoundingBox(object):
    def test_basic_box(self):
        bbox1 = BoundingBox(0, 0, 500, 500)
        assert bbox1.x1 == 0
        assert bbox1.y1 == 0
        assert bbox1.x2 == 499
        assert bbox1.y2 == 499
        assert bbox1.h == 500
        assert bbox1.w == 500
        
    def test_nonbasic_box(self):
        bbox2 = BoundingBox(24, 48, 64, 96)

        assert bbox2.x1 == 24
        assert bbox2.y1 == 48
        assert bbox2.x2 == 87
        assert bbox2.y2 == 143
        assert bbox2.h == 96
        assert bbox2.w == 64
