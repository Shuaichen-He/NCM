# -*- coding: UTF-8 -*-
"""
@file:test_gauss_hermite_8.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_integration_04.gauss_hermite_int import GaussHermiteIntegration


fun = lambda x: np.exp(-x ** 2) * x ** 2
ghi = GaussHermiteIntegration(fun, [-np.infty, np.infty], 10)
ghi.cal_int()
print("埃尔米特多项式零点：", ghi.zero_points)
print("插值型求积系数：", ghi.A_k)
print("积分值：", ghi.int_value, "误差：", np.sqrt(np.pi) / 2 - ghi.int_value)
