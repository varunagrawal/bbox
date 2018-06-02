import numpy as np


def jaccard_index(a, b):
    """
    Computes the Intersection over Union of two sets of bounding boxes.
    Also known as IoU. 
    """
    xA = np.maximum(a.x1, b.x1.T)
    yA = np.maximum(a.y1, b.y1.T)
    xB = np.minimum(a.x2, b.x2.T)
    yB = np.minimum(a.y2, b.y2.T)

    intersection = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

    a_area = a.width * a.height
    b_area = b.width * b.height

    iou = intersection / (a_area + b_area.T - intersection)

    return iou
