"""
Useful functions to deal with 3D geometry
"""
import numpy as np
from bbox import BBox3D


def plane(a, b, c):
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


def polygon_area(polygon):
    """
    Get the area of a polygon which is represented by a 2D array of points.
    Area is computed using the Shoelace Algorithm.
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
