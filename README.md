# bbox

`bbox` a Python library that is intended to ease the use of 2D and 3D bounding boxes in areas such as Object Detection by providing a set of flexible primitives and functions that are intuitive and easy to use out of the box.

[![Build Status](https://travis-ci.org/varunagrawal/bbox.svg?branch=master)](https://travis-ci.org/varunagrawal/bbox)
[![codecov](https://codecov.io/gh/varunagrawal/bbox/branch/master/graph/badge.svg)](https://codecov.io/gh/varunagrawal/bbox)


[![PyPI version](https://badge.fury.io/py/bbox.svg)](https://badge.fury.io/py/bbox)
![PyPI format](https://img.shields.io/pypi/format/bbox.svg)
![](https://img.shields.io/pypi/status/bbox.svg)
![](https://img.shields.io/pypi/pyversions/bbox.svg)


![](https://img.shields.io/pypi/l/bbox.svg)
[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/varunagrawal)


## Features

### 2D Bounding Box

Easily work with bounding boxes using a simple class that abstracts and maintains various attributes.

```python
from bbox import BBox2D

# x, y, w, h
box = BBox2D([0, 0, 32, 32])

# equivalently, in (x1, y1, x2, y2) (aka two point format), we can use
box = BBox2D([0, 0, 31, 31], mode=XYXY)

print(box.x1, box.y1)  # -> 0 0
print(box.x2, box.y2)  # -> 31 31
print(box.height, box.width)  # -> 32 32

# Syntatic sugar for height and width
print(box.h, box.w)  # -> 32 32
```
### Sequence of 2D bounding boxes

Most tasks involve dealing with multiple bounding boxes. This can also be handled conveniently with the `BBox2DList` class.

```python
bbl = BBox2DList(np.random.randint(10, 4),
                 mode=XYWH)
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
