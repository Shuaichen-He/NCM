# -*- coding: UTF-8 -*-
"""
@file:test_gauss_legendre_5.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_integration_04.gauss_legendre_int import GaussLegendreIntegration


def fun(x):
    return np.exp(-x) * np.sin(x)


exact_val = (1 - np.exp(-8) * (np.cos(8) + np.sin(8))) / 2  # 积分精确值
gli = GaussLegendreIntegration(fun, [0, 8], 15)
gli.fit_int()
print("积分近似值：", gli.int_value, "误差：", exact_val - gli.int_value)
print("勒让德多项式零点：", gli.zero_points)
print("插值型求积系数：", gli.A_k)
