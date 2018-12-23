# bbox

`bbox` a Python library that is intended to ease the use of 2D and 3D bounding boxes in areas such as Object Detection by providing a set of flexible primitives and functions that are intuitive and easy to use out of the box.

## Features

### 2D Bounding Box

Easily work with bounding boxes using a simple class that abstracts and maintains various attributes.

```python
from bbox import BBox2D

# x, y, w, h
box = BBox2D([0, 0, 32, 32])

# equivalently, in (x1, y1, x2, y2) (aka two point format), we can use
box = BBox2D([0, 0, 31, 31], two_point=True)

print(bbox.x1, bbox.y1)  # -> 0 0
print(bbox.x2, bbox.y2)  # -> 31 31
print(bbox.height, bbox.width)  # -> 32 32

# Syntatic sugar for height and width
print(bbox.h, bbox.w)  # -> 32 32
```
### Sequence of 2D bounding boxes

Most tasks involve dealing with multiple bounding boxes. This can also be handled conveniently with the `BBox2DList` class.

```python
bbl = BBox2DList(np.random.randint(10, 4),
                 two_point=False)
```

The above snippet creates a list of 10 bounding boxes neatly abstracted into a convenient object.

### Non-maximum Suppression

Need to perform non-maximum suppression? It is as easy as a single function call.
```python
from bbox.utils import nms

# bbl -> BBox2DList
# scores -> list/ndarray of confidence scores
new_boxes = nms(bbl, scores)
```

### Intersection over Union (Jaccard Index)

The Jaccard Index or IoU is a very useful metric for finding similarities between bounding boxes. `bbox` provides native support for this.

```python
from bbox.metrics import jaccard_index_2d

box1 = BBox2D([0, 0, 32, 32])
box2 = BBox2D([10, 12, 32, 46])

iou = jaccard_index_2d(box1, box2)
```

We can even use the Jaccard Index to compute a distance metric between boxes as a distance matrix:

```python
from bbox.metrics import multi_jaccard_index_2d

dist = 1 - multi_jaccard_index_2d(bbl, bbl)
```

### 3D Bounding Box

`bbox` also support 3D bounding boxes, providing convenience methods and attributes for working with them.
