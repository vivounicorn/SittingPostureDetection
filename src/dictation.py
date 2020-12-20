# coding:utf-8

import os, time
from src.tts import TTS
from src.logger import Logger
from src.config import Config

logger = Logger(__name__).getLogger()


class Dictation(object):

    def __init__(self):
        self.init_success = False
        cfg_path = os.getcwd() + '/config/cfg.ini'
        if not os.path.exists(cfg_path):
            raise IOError("The configuration file was not found.")

        self.cfg = Config(cfg_path)

        self.words_path = self.cfg.dictation_words_path()
        self.duration = self.cfg.dictation_duration()
        self.frequency = self.cfg.dictation_frequency()
        self.opening_remarks = "熙熙，现在开始听写词语，保持好坐姿，我们马上开始。每个词语读%d遍，每次停顿%d秒。"
        self.concluding_remarks = "听写结束，表扬熙熙。"
        self.tts = TTS()

    def read_out(self):

        self.tts.voice(self.opening_remarks % (self.frequency,self.duration))
        if not os.path.exists(self.words_path):
            logger.error("file [%s] is not exists." % self.words_path)
            return

        with open(self.words_path, 'rt') as f:
            for line in f:
                if not len(line.strip()) or line.startswith('#'):
                    continue
                for i in range(self.frequency):
                    self.tts.voice(line)
                    time.sleep(self.duration)

        self.tts.voice(self.concluding_remarks)
