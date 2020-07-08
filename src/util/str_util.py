# encoding=utf-8
import re


def re_digit(text):
    '''
    去除字符串里面的数字和#号
    :param text:
    :return:
    '''
    pattern = '[\d]+'
    return re.sub(pattern, '', text).replace('#', '')