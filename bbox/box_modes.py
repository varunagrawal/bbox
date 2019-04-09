from enum import Enum


class BoxMode(Enum):
    XYWH = 0
    XYXY = 1


XYWH = BoxMode.XYWH.value
XYXY = BoxMode.XYXY.value