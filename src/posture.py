import json
import logging
import math

from src.point import Point2D
from src.tts import TTS

# [{"keypoints": [203.26, 135.15, 0.51, 204.47, 125.27, 0.51, -6.0, -2.0, 0.0, 242.72, 107.57, 0.46, -6.0, -2.0, 0.0,
# 319.87, 125.81, 0.42, 282.44, 112.03, 0.28, 237.12, 177.62, 0.86, 220.59, 141.05, 0.29, 183.56, 157.62, 0.4,
# 171.85, 148.71, 0.4, 368.2, 253.97, 0.26, 336.76, 241.86, 0.19, 223.72, 312.84, 0.25, 234.99, 258.77, 0.15, -6.0,
# -2.0, 0.0, -6.0, -2.0, 0.0], "bbox": [154.99, 89.9, 240.21, 247.29], "score": 0.38, "category_id": 1}]

'''
0 nose
1 left_eye
2 right_eye
3 left_ear
4 right_ear
5 left_shoulder
6 right_shoulder
7 left_elbow
8 right_elbow
9 left_wrist
10 right_wrist
11 left_hip
12 right_hip
13 left_knee
14 right_knee
15 left_ankle
16 right_ankle
'''
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Posture(object):

    def __init__(self, jstr: str):

        j = json.loads(jstr)
        if len(j) == 0:
            logger.info(jstr)
        else:
            self.keypoints = j[0]['keypoints']
            self.decode_json()
        self.tts = TTS()

    def decode_json(self):
        self.nose = Point2D('nose', self.keypoints[0], self.keypoints[1], self.keypoints[2])
        self.left_eye = Point2D('left_eye', self.keypoints[3], self.keypoints[4], self.keypoints[5])
        self.right_eye = Point2D('right_eye', self.keypoints[6], self.keypoints[7], self.keypoints[8])
        self.left_ear = Point2D('left_ear', self.keypoints[9], self.keypoints[10], self.keypoints[11])
        self.right_ear = Point2D('right_ear', self.keypoints[12], self.keypoints[13], self.keypoints[14])
        self.left_shoulder = Point2D('left_shoulder', self.keypoints[15], self.keypoints[16], self.keypoints[17])
        self.right_shoulder = Point2D('right_shoulder', self.keypoints[18], self.keypoints[19], self.keypoints[20])
        self.left_elbow = Point2D('left_elbow', self.keypoints[21], self.keypoints[22], self.keypoints[23])
        self.right_elbow = Point2D('right_elbow', self.keypoints[24], self.keypoints[25], self.keypoints[26])
        self.left_wrist = Point2D('left_wrist', self.keypoints[27], self.keypoints[28], self.keypoints[29])
        self.right_wrist = Point2D('right_wrist', self.keypoints[30], self.keypoints[31], self.keypoints[32])
        self.left_hip = Point2D('left_hip', self.keypoints[33], self.keypoints[34], self.keypoints[35])
        self.right_hip = Point2D('right_hip', self.keypoints[36], self.keypoints[37], self.keypoints[38])
        self.left_knee = Point2D('left_knee', self.keypoints[39], self.keypoints[40], self.keypoints[41])
        self.right_knee = Point2D('right_knee', self.keypoints[41], self.keypoints[43], self.keypoints[44])
        self.left_ankle = Point2D('left_ankle', self.keypoints[44], self.keypoints[46], self.keypoints[47])
        self.right_ankle = Point2D('right_ankle', self.keypoints[48], self.keypoints[49], self.keypoints[50])

    def shoulder_hip_knee_left(self):
        return self.left_hip.angle(self.left_shoulder, self.left_knee)

    def shoulder_hip_knee_right(self):
        return self.right_hip.angle(self.right_shoulder, self.right_knee)

    def ear_shoulder_hip_left(self):
        return self.left_shoulder.angle(self.left_ear, self.left_hip)

    def ear_shoulder_hip_right(self):
        return self.right_shoulder.angle(self.right_ear, self.right_hip)

    def detect(self, is_right=False):
        if is_right:
            angle1 = self.shoulder_hip_knee_right()
            angle2 = self.ear_shoulder_hip_right()
        else:
            angle1 = self.shoulder_hip_knee_left()
            angle2 = self.ear_shoulder_hip_left()

        if math.fabs(111-angle1) > 10 or math.fabs(165-angle2) > 10:
            logger.info(angle1, angle2)
            self.tts.voice('熙熙，坐直腰，伸直脖子，别趴着写字。')
