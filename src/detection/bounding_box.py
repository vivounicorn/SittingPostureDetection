# coding:utf-8

import json
from src.utils.logger import Logger

logger = Logger(__name__).getLogger()


class BoundingBox(object):

    def __init__(self, jstr: str):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        j = json.loads(jstr)
        if len(j) == 0:
            logger.info(jstr)
        else:
            bbox = j[0]['bbox']
            self.x = float(bbox[0])
            self.y = float(bbox[1])
            self.w = float(bbox[2])
            self.h = float(bbox[3])

    def __str__(self):
        return "%.2f,%.2f,%.2f,%.2f" % (self.x, self.y, self.w, self.h)
