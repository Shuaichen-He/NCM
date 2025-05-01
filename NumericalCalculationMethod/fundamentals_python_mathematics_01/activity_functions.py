# -*- coding: UTF-8 -*-
"""
@file_name: activity_functions.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import numpy as np
from scipy.special import expit


def activity_functions(act_type):
    """
    定义局部函数，使用函数作为返回值，定义两个激活函数
    :param act_type: 激活函数类型
    :return:
    """

    def sigmoid(x):
        x = np.array(x, dtype=np.float64)
        return expit(x)

    def diff_sigmoid(x):
        return x * (1 - x)

    def tanh(x):
        return np.tanh(x)

    def diff_tanh(x):
        return 1 - tanh(x) ** 2

    if act_type == "sigmoid":
        return sigmoid, diff_sigmoid
    elif act_type == "tanh":
        return tanh, diff_tanh
    else:
        raise AttributeError("损失函数选择错误。。。")