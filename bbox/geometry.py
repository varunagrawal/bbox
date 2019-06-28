"""
Useful functions to deal with 3D geometry
"""

# pylint: disable=invalid-name,missing-docstring,invalid-unary-operand-type,no-else-return

import numpy as np


def get_plane(a, b, c):
    """
    Get plane equation from 3 points.
    Returns the coefficients of `ax + by + cz + d = 0`
    """
    ab = b - a
    ac = c - a

    x = np.cross(ab, ac)
    d = -np.dot(x, a)
    pl = np.hstack((x, d))
    return pl


def point_plane_dist(pt, plane, signed=False):
    """
    Get the signed distance from a point `pt` to a plane `plane`.
    Reference: http://mathworld.wolfram.com/Point-PlaneDistance.html

    Plane is of the format [A, B, C, D], where the plane equation is Ax+By+Cz+D=0
    Point is of the form [x, y, z]
    `signed` flag indicates whether to return signed distance.
    """
    v = plane[0:3]
    dist = (np.dot(v, pt) + plane[3]) / np.linalg.norm(v)

    if signed:
        return dist
    else:
        return np.abs(dist)


def edges_of(vertices):
    """
    Return the vectors for the edges of the polygon defined by `vertices`.

    Args:
        vertices: list of vertices of the polygon.
    """
    edges = []
    N = len(vertices)

    for i in range(N):
        edge = vertices[(i + 1) % N] - vertices[i]
        edges.append(edge)

    return edges


def orthogonal(v):
    """
    Return a 90 degree clockwise rotation of the vector `v`.

    Args:
        v: 2D array representing a vector.
    """
    return np.array([-v[1], v[0]])


def is_separating_axis(o, p1, p2):
    """
    Return True and the push vector if `o` is a separating axis of `p1` and `p2`.
    Otherwise, return False and None.

    Args:
        o: 2D array representing a vector.
        p1: 2D array of points representing a polygon.
        p2: 2D array of points representing a polygon.
    """
    min1, max1 = float('+inf'), float('-inf')
    min2, max2 = float('+inf'), float('-inf')

    for v in p1:
        projection = np.dot(v, o)

        min1 = min(min1, projection)
        max1 = max(max1, projection)

    for v in p2:
        projection = np.dot(v, o)

        min2 = min(min2, projection)
        max2 = max(max2, projection)

    if max1 >= min2 and max2 >= min1:
        d = min(max2 - min1, max1 - min2)
        # push a bit more than needed so the shapes do not overlap in future
        # tests due to float precision
        d_over_o_squared = d/np.dot(o, o) + 1e-10
        pv = d_over_o_squared*o
        return False, pv
    else:
        return True, None


def polygon_collision(p1, p2):
    """
    Return True if the shapes collide. Otherwise, return False.

    p1 and p2 are np.arrays, the vertices of the polygons in the
    counterclockwise direction.

    Source: https://hackmd.io/s/ryFmIZrsl

    Args:
        p1: 2D array of points representing a polygon.
        p2: 2D array of points representing a polygon.
    """
    edges = edges_of(p1)
    edges += edges_of(p2)
    orthogonals = [orthogonal(e) for e in edges]

    push_vectors = []
    for o in orthogonals:
        separates, pv = is_separating_axis(o, p1, p2)

        if separates:
            # they do not collide and there is no push vector
            return False
        else:
            push_vectors.append(pv)

    return True


def polygon_area(polygon):
    """
    Get the area of a polygon which is represented by a 2D array of points.
    Area is computed using the Shoelace Algorithm.

    Args:
        polygon: 2D array of points.
    """
    x = polygon[:, 0]
    y = polygon[:, 1]
    area = (np.dot(x, np.roll(y, -1)) -
            np.dot(np.roll(x, -1), y))
    return np.abs(area)/2


def polygon_intersection(poly1, poly2):
    """
    Use the Sutherland-Hodgman algorithm to compute the intersection of 2 convex polygons.
    """
    def line_intersection(e1, e2, s, e):
        dc = e1 - e2
        dp = s - e
        n1 = np.cross(e1, e2)
        n2 = np.cross(s, e)
        n3 = 1.0 / (np.cross(dc, dp))
        return np.array([(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3])

    def is_inside_edge(p, e1, e2):
        """Return True if e is inside edge (e1, e2)"""
        return np.cross(e2-e1, p-e1) >= 0

    output_list = poly1
    # e1 and e2 are the edge vertices for each edge in the clipping polygon
    e1 = poly2[-1]

    for e2 in poly2:
        input_list = output_list
        output_list = []
        s = input_list[-1]

        for e in input_list:
            if is_inside_edge(e, e1, e2):
                # if s in not inside edge (e1, e2)
                if not is_inside_edge(s, e1, e2):
                    # line intersects edge hence we compute intersection point
                    output_list.append(line_intersection(e1, e2, s, e))
                output_list.append(e)
            # is s inside edge (e1, e2)
            elif is_inside_edge(s, e1, e2):
                output_list.append(line_intersection(e1, e2, s, e))

            s = e
        e1 = e2

    return np.array(output_list)
