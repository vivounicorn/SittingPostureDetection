import os
from aip import AipSpeech
from playsound import playsound


class TTS(object):

    def __init__(self):
        self.in_path = "/tmp/audio"
        try:
            os.mkdir(self.in_path)
        except FileExistsError:
            pass

        # 你的 APPID AK SK
        APP_ID = '14371003'
        API_KEY = '6ACWsLOg3b7OyGUKGfHZfbXa'
        SECRET_KEY = 'nA6WP1d05qBoUYqxplNAV1inf8IHGwj9'

        self.client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

    def voice(self, sentenses=''):

        result = self.client.synthesis(sentenses, 'zh', 1,
                                       {'vol': 5,  # 音量，取值0-15，默认为5中音量
                                        'per': 4,  # 发音人选择, 0为女声，1为男声，3为情感合成-度逍遥，4为情感合成-度丫丫，默认为普通女
                                        'spd': 5,  # 语速，取值0-9，默认为5中语速
                                        'pit': 5,  # 音调，取值0-9，默认为5中语调
                                        'aue': 6
                                        }
                                       )

        if not isinstance(result, dict):
            with open(self.in_path + 'tmp.mp3', 'wb') as f:
                f.write(result)
        f.close()

        playsound(self.in_path + 'tmp.mp3')
