import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def cuboid_data(o, size=(1, 1, 1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1], o[1], o[1]],
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
    z = [[o[2], o[2], o[2], o[2], o[2]],
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]
    return np.array(x), np.array(y), np.array(z)


def plotCubeAt(pos=(0, 0, 0), size=(1, 1, 1), ax=None, **kwargs):
    # Plotting a cube element at position pos
    if ax != None:
        X, Y, Z = cuboid_data(pos, size)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, **kwargs)


from bbox import BBox3D

bbox_a = BBox3D(x=332.5, y=85.0, z=205.5, length=125, width=74, height=45)
bbox_b = BBox3D(x=332.5, y=144.0, z=366.5, length=113, width=44, height=65)

print(bbox_a.center)
positions = [bbox_a.p1, bbox_b.p1]
sizes = [(bbox_a.l, bbox_a.w, bbox_a.h), (bbox_b.l, bbox_b.w, bbox_b.h)]
colors = ["crimson", "limegreen"]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax.set_aspect('equal')

for p, s, c in zip(positions, sizes, colors):
    plotCubeAt(pos=p, size=s, ax=ax, color=c)

plt.show()
