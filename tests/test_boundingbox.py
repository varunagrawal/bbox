from bbox import BoundingBox, BoundingBoxList

def main():
    bbox1 = BoundingBox(0, 0, 500, 500)
    bbox2 = BoundingBox(24, 48, 64, 96)

    assert bbox1.x1 == 0
    assert bbox1.y1 == 0
    assert bbox1.x2 == 500
    assert bbox1.y2 == 500
    assert bbox1.h == 500, "[BoundingBox] Invalid height computation"
    assert bbox2.x1 == 24
    assert bbox2.y1 == 48
    assert bbox2.x2 == 88
    assert bbox2.y2 == 144

    l = BoundingBoxList([bbox1, bbox2])
    assert len(l) == 2
    assert l.shape == (2, 4), "[BoundingBoxList] Incorrect shape of list array"
    

if __name__ == "__main__":
    main()
