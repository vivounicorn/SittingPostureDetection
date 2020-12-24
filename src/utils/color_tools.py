# -*- coding:utf-8 -*-

import random


def random_rgb2hex():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    color = '#'
    color += str(hex(r))[-2:].replace('x', '0').upper()
    color += str(hex(g))[-2:].replace('x', '0').upper()
    color += str(hex(b))[-2:].replace('x', '0').upper()
    return color


def rgb2hex(rgb='123,235,60'):
    RGB = rgb.split(',')
    color = '#'
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color


def hex2rgb(hex_cor='#87FFE6'):
    r = int(hex_cor[1:3], 16)
    g = int(hex_cor[3:5], 16)
    b = int(hex_cor[5:7], 16)
    rgb = '%d,%d,%d' % (r, g, b)
    return rgb
