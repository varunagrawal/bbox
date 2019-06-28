""""""

# pylint: disable=invalid-name,missing-docstring

import numpy as np
#TODO Convert tests to Pytests
import pytest
from bbox import BBox3D
from bbox.geometry import get_plane, point_plane_dist, polygon_area, polygon_intersection, \
    polygon_collision, edges_of, orthogonal, is_separating_axis


def clip(subject_polygon, clip_polygon):
    """
    Naive implementation picked up from Rosetta Code.
    Only used for testing purposes.
    """
    def inside(p):
        return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

    def compute_intersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

    output_list = subject_polygon
    cp1 = clip_polygon[-1]

    for clip_vertex in clip_polygon:
        cp2 = clip_vertex
        input_list = output_list
        output_list = []
        s = input_list[-1]

        for subject_vertex in input_list:
            e = subject_vertex
            if inside(e):
                if not inside(s):
                    output_list.append(compute_intersection())
                output_list.append(e)
            elif inside(s):
                output_list.append(compute_intersection())
            s = e
        cp1 = cp2
    return output_list


def test_plane():
    # define the 3 points
    a = np.array([1, 1, 1])
    b = np.array([-1, 1, 0])
    c = np.array([2, 0, 3])
    assert np.array_equal(get_plane(a, b, c), np.array([-1, 3, 2, -4]))


def test_point_plane_dist():
    pt = np.array([2, 8, 5])
    plane = np.array([1, -2, -2, -1])
    assert point_plane_dist(pt, plane) == 25/3
    assert point_plane_dist(pt, plane, signed=True) == -25/3


def test_polygon_area():
    polygon = np.array([[-3, -2], [-1, 4], [6, 1], [3, 10], [-4, 9]])
    assert polygon_area(polygon) == 60


def test_polygon_intersection():
    a = BBox3D(0.5, 0.5, 0.5, 1, 1, 1)
    b = BBox3D(1, 1, 1, 1, 1, 1)

    i1 = clip(a.p[0: 4, 0: 2], b.p[0: 4, 0: 2])
    i1 = np.array(i1)
    i2 = polygon_intersection(a.p[0: 4, 0: 2], b.p[0: 4, 0: 2])
    assert np.array_equal(i1, i2)


# Ensure 100% test coverage
def test_is_separating_axis():
    # randomly generated values
    a = BBox3D(0.9, 0.5, 0.5, 1, 1, 1)
    b = BBox3D(0.1, 0.1, 0.1, 0.1, 1, 1)

    p1, p2 = a.p[0:4, 0:2], b.p[0:4, 0:2]
    edges = edges_of(p1)
    edges += edges_of(p2)

    # positive case
    separates, pv = is_separating_axis(orthogonal(edges[0]), p1, p2)
    assert not separates and pv is not None

    separates, pv = is_separating_axis(orthogonal(edges[1]), p1, p2)
    assert separates and pv is None


def test_polygon_no_collision():
    # randomly generated values
    a = BBox3D(0.9, 0.5, 0.5, 1, 1, 1)
    b = BBox3D(0.1, 0.1, 0.1, 0.1, 1, 1)
    p1, p2 = a.p[0:4, 0:2], b.p[0:4, 0:2]

    assert not polygon_collision(p1, p2)
