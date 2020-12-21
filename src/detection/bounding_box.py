# coding:utf-8

import math
from src.utils.logger import Logger

logger = Logger(__name__).getLogger()


class BoundingBox(object):

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h