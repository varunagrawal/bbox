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


def point_plane_dist(pt, plane):
    """
    Get the signed distance from a point `pt` to a plane `plane`.
    Reference: http://mathworld.wolfram.com/Point-PlaneDistance.html
    """
    v = plane[0:3]
    dist = (np.dot(v, pt) + plane[3]) / np.linalg.norm(v)
    return dist


def polygon_area(polygon):
    """
    Get the area of a polygon which is represented by a 2D array of points.
    Area is computed using the Shoelace Algorithm.
    """
    x = polygon[:, 0]
    y = polygon[:, 1]
    area = (np.dot(x[:-1], np.roll(y, -1)[:-1]) -
            np.dot(np.roll(x, -1)[:-1], y[:-1])) / 2
    return area


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

    output_list = poly1
    # e1 and e2 are the edge vertices for each edge in the clipping polygon
    e1 = poly2[-1]

    for e2 in poly2:
        input_list = output_list
        output_list = []
        s = input_list[-1]

        for e in input_list:
            # if e is inside edge (e1, e2)
            if np.cross(e2-e1, e-e1) > 0:
                # if s in not inside edge (e1, e2)
                if not np.cross(e2-e1, s-e1) > 0:
                    # line intersects edge hence we compute intersection point
                    output_list.append(line_intersection(e1, e2, s, e))
                output_list.append(e)
            # is s inside edge (e1, e2)
            elif np.cross(e2-e1, s-e1) > 0:
                output_list.append(line_intersection(e1, e2, s, e))

            s = e
        e1 = e2

    return np.array(output_list)
