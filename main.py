# coding:utf-8

from src.custom_estimator import CustomFormatter
from src.posture import Posture
from openpifpaf import predict
from src.tts import TTS

if __name__ == '__main__':
    e = CustomFormatter(cfg_path='/home/zhanglei/Gitlab/SittingPostureDetection/config/cfg.ini')
    e.run()
