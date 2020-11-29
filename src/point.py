import math
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Point2D(object):

    def __init__(self, name='', x=0.0, y=0.0, score=0.0):
        self.name = name
        self.X = x
        self.Y = y
        self.score = score
        self.EPS = 0.00001

    def __str__(self):
        return "(%s %.2f, %.2f)" % (self.name, self.X, self.Y)

    def distence(self, p) -> float:
        return math.sqrt((self.X - p.X) ** 2 + (self.Y - p.Y) ** 2)

    def cosine_law(self, pt_up, pt_down) -> float:
        ab = self.distence(pt_up)
        bc = self.distence(pt_down)
        ac = pt_up.distence(pt_down)

        logger.debug(str(self),str(pt_up), str(pt_down))
        if math.fabs(ab) < self.EPS or math.fabs(bc-self.EPS):
            return 0
        else:
            return (ab ** 2 + bc ** 2 - ac ** 2) / (2 * ab * bc)

    def angle(self, pt_up, pt_down) -> float:
        try:
            return math.acos(self.cosine_law(pt_up, pt_down)) * 180 / math.pi
        except:
            pass
