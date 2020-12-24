#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from configparser import ConfigParser


class Config(object):

    def __init__(self, file_path):

        self.config = ConfigParser(comment_prefixes='/', allow_no_value=True)
        if not os.path.exists(file_path):
            raise IOError("Can't read file(%s)" % file_path)

        self.config.read(file_path)
        self.file_path = file_path
        if not self.config.has_section("parameters"):
            raise IOError("Can't read section parameters")
        if not self.config.has_section("voice_text"):
            raise IOError("Can't read section voice_text")
        if not self.config.has_section("file_path"):
            raise IOError("Can't read section file_path")

    def skip_frame(self):
        if self.config.has_option("parameters", 'skip_frame'):
            return self.config.getint("parameters", 'skip_frame')
        return 10

    def shoulder_waist_knee_angle(self):
        if self.config.has_option("parameters", 'shoulder_waist_knee_angle'):
            return self.config.getfloat("parameters", 'shoulder_waist_knee_angle')
        return 0.0

    def ear_shoulder_waist_angle(self):
        if self.config.has_option("parameters", 'ear_shoulder_waist_angle'):
            return self.config.getfloat("parameters", 'ear_shoulder_waist_angle')
        return 0.0

    def set_shoulder_waist_knee_angle(self, val):
        if self.config.has_option("parameters", 'shoulder_waist_knee_angle'):
            self.config.set("parameters", 'shoulder_waist_knee_angle', val)

    def set_ear_shoulder_waist_angle(self, val):
        if self.config.has_option("parameters", 'ear_shoulder_waist_angle'):
            self.config.set("parameters", 'ear_shoulder_waist_angle', val)

    def shoulder_waist_knee_angle_th(self):
        if self.config.has_option("parameters", 'shoulder_waist_knee_angle_th'):
            return self.config.getint("parameters", 'shoulder_waist_knee_angle_th')
        return 0

    def bbox(self):
        orgi = (0.0, 0.0, 0.0, 0.0)
        if self.config.has_option("parameters", 'bbox'):
            pos = self.config.get("parameters", 'bbox')
            box = pos.split(',')
            if len(box) == 4:
                p = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                return p
        return orgi

    def set_bbox(self, val):
        if self.config.has_option("parameters", 'bbox'):
            self.config.set("parameters", 'bbox', val)

    def ear_shoulder_waist_angle_th(self):
        if self.config.has_option("parameters", 'ear_shoulder_waist_angle_th'):
            return self.config.getint("parameters", 'ear_shoulder_waist_angle_th')
        return 0

    def voice_waist_list(self):
        if self.config.has_option("voice_text", 'voice_waist_list'):
            return self.config.get("voice_text", 'voice_waist_list').split(',')
        return []

    def voice_ear_list(self):
        if self.config.has_option("voice_text", 'voice_ear_list'):
            return self.config.get("voice_text", 'voice_ear_list').split(',')
        return []

    def voice_encourage(self):
        if self.config.has_option("voice_text", 'voice_encourage'):
            return self.config.get("voice_text", 'voice_encourage')
        return ''

    def voice_no_posture(self):
        if self.config.has_option("voice_text", 'voice_no_posture'):
            return self.config.get("voice_text", 'voice_no_posture')
        return ''

    def camera_calibration_path(self):
        if self.config.has_option("file_path", 'camera_calibration_path'):
            return self.config.get("file_path", 'camera_calibration_path')
        return ''

    def dictation_words_path(self):
        if self.config.has_option("file_path", 'dictation_words_path'):
            return self.config.get("file_path", 'dictation_words_path')
        return ''

    def dictation_duration(self):
        if self.config.has_option("parameters", 'dictation_duration'):
            return self.config.getint("parameters", 'dictation_duration')
        return ''

    def dictation_frequency(self):
        if self.config.has_option("parameters", 'dictation_frequency'):
            return self.config.getint("parameters", 'dictation_frequency')
        return ''

    def yolov5s_path(self):
        if self.config.has_option("file_path", 'yolov5s_path'):
            return self.config.get("file_path", 'yolov5s_path')
        return ''

    def flush(self):
        with open(self.file_path, "w+") as f:
            self.config.write(f)
