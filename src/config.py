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

    def camera_calibration_path(self):
        if self.config.has_option("file_path", 'camera_calibration_path'):
            return self.config.get("file_path", 'camera_calibration_path')
        return ''

    def flush(self):
        with open(self.file_path, "w+") as f:
            self.config.write(f)
