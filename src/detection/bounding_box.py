# coding:utf-8

import json
from src.utils.logger import Logger

logger = Logger(__name__).getLogger()


class BoundingBox(object):

    def __init__(self, jstr):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        if isinstance(jstr, str):
            j = json.loads(jstr)
            if len(j) == 0:
                logger.info(jstr)
            else:
                bbox = j[0]['bbox']
                self.x = int(bbox[0])
                self.y = int(bbox[1])
                self.w = int(bbox[2])
                self.h = int(bbox[3])
        elif isinstance(jstr, tuple):
            self.x = jstr[0]
            self.y = jstr[1]
            self.w = jstr[2]
            self.h = jstr[3]

    def __str__(self):
        return "%.2f,%.2f,%.2f,%.2f" % (self.x, self.y, self.w, self.h)

    def get_crop_tuple(self):
        return self.x, self.y, self.x+self.w, self.y+self.h
