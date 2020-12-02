# -*- coding:utf-8 -*-

import logging
import logging.handlers
import time
import os


class Logger(logging.Logger):

    def __init__(self, name: str, logger=None, log_cate='search'):

        super().__init__(name)
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        self.log_time = time.strftime("%Y_%m_%d")
        file_dir = os.getcwd() + '/logs'
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        self.log_path = file_dir
        self.log_name = self.log_path + "/" + log_cate + "." + self.log_time + '.log'

        # fh = logging.FileHandler(self.log_name, 'a', encoding='utf-8')  # 这个是python3的
        fh = logging.handlers.TimedRotatingFileHandler(self.log_name, 'D', 1, 30)
        fh.suffix = "%Y%m%d-%H%M.log"
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '[%(asctime)s] %(filename)s->%(funcName)s line:%(lineno)d [%(levelname)s]%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        fh.close()
        ch.close()

    def getLogger(self):
        return self.logger
