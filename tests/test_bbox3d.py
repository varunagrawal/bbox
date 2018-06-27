from bbox import BBox3D
import numpy as np
import quaternion
from PIL import Image, ImageDraw


K = np.array([[1406.3359, 0.0, 966.366034, 0.0],
              [0.0, 1408.94297, 607.479746, 0.0],
              [0.0, 0.0, 1.0, 0.0]])

R = np.array([[0.50478576,  0.86323317, -0.00445338],
              [-0.00422247, -0.00268975, -0.99998747],
              [-0.86323433,  0.50479824,  0.00228723]])

t = np.array([[-0.75116634], [1.35776453], [0.87137971]])

cuboid = {
    'center': {
        'x': -49.19743041908411,
        'y': 12.38666074615689,
        'z': 0.782056864653507
    },
    'dimensions': {
        'height': 1.9422248281533563,
        'length': 5.340892485711914,
        'width': 2.457703972075464
    },
    'rotation': {
        'w': 0.9997472337219893,
        'x': 0.0,
        'y': 0.0,
        'z': 0.022482630300529462
    }
}

center = cuboid['center']
dim = cuboid['dimensions']
rotation = cuboid['rotation']

box = BBox3D(center['x'], center['y'], center['z'], dim['length'], dim['width'],
             dim['height'], rotation['x'], rotation['y'], rotation['z'], rotation['w'], center=True)

c = box.center


def project(p, K, R, t):
    E = np.eye(4)
    E[0:3, 0:3] = R
    E[0:3, 3:4] = t

    u = K @ E @ p
    # print("projection", u)
    return u[0:2] / u[2]


print(box.p1)
print(box.p2)
print(box.p3)
print(box.p4)
print(box.p5)
print(box.p6)
print(box.p7)
print(box.p8)
print("\n\n\n")

u1 = project(box.p1, K, R, t)
print(u1)
u2 = project(box.p2, K, R, t)
print(u2)
u3 = project(box.p3, K, R, t)
print(u3)
u4 = project(box.p4, K, R, t)
print(u4)
u5 = project(box.p5, K, R, t)
print(u5)
u6 = project(box.p6, K, R, t)
print(u6)
u7 = project(box.p7, K, R, t)
print(u7)
u8 = project(box.p8, K, R, t)
print(u8)

# c = project(c, K, R, t)
# print("c", c)

print("\n\n\n")


def draw_cuboid(img, p):
    draw = ImageDraw.Draw(img)
    color = tuple(np.random.choice(range(256), size=3))

    draw.line([p[0][0], p[0][1], p[1][0], p[1][1]], fill=color, width=2)
    draw.line([p[1][0], p[1][1], p[5][0], p[5][1]], fill=color, width=2)
    draw.line([p[5][0], p[5][1], p[4][0], p[4][1]], fill=color, width=2)
    draw.line([p[4][0], p[4][1], p[0][0], p[0][1]], fill=color, width=2)

    draw.line([p[3][0], p[3][1], p[2][0], p[2][1]], fill=color, width=2)
    draw.line([p[2][0], p[2][1], p[6][0], p[6][1]], fill=color, width=2)
    draw.line([p[6][0], p[6][1], p[7][0], p[7][1]], fill=color, width=2)
    draw.line([p[7][0], p[7][1], p[3][0], p[3][1]], fill=color, width=2)

    draw.line([p[0][0], p[0][1], p[3][0], p[3][1]], fill=color, width=2)
    draw.line([p[1][0], p[1][1], p[2][0], p[2][1]], fill=color, width=2)
    draw.line([p[5][0], p[5][1], p[6][0], p[6][1]], fill=color, width=2)
    draw.line([p[4][0], p[4][1], p[7][0], p[7][1]], fill=color, width=2)
    return img


img = Image.open('image_raw_ring_rear_left_315968023469083760.png')
img = draw_cuboid(img, [u1, u2, u3, u4, u5, u6, u7, u8])
img.show()


def get_cuboid_rect(cuboid):
    size = cuboid['dimensions']
    h, l, w = size['height'], size['length'], size['width']  # z,x,y

    center = cuboid['center']

    q = np.quaternion(cuboid['rotation']['w'], cuboid['rotation']
                      ['x'], cuboid['rotation']['y'], cuboid['rotation']['z'])
    rotMat = quaternion.as_rotation_matrix(q)
    trackletBox = np.array([
        [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [-h/2,  -h/2,  -h/2,  -h/2,  h/2,   h/2,   h/2,   h/2]
    ])

    cornerPosInVelo = np.dot(rotMat, trackletBox)
    cornerPosInVelo = cornerPosInVelo + \
        np.tile([center['x'], center['y'], center['z']], (8, 1)).T
    frame_rects = cornerPosInVelo
    return frame_rects


print(get_cuboid_rect(cuboid).T)
