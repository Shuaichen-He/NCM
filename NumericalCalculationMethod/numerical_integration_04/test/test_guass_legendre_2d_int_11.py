# -*- coding: UTF-8 -*-
"""
@file_name: test_guass_legendre_2d_int_11.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_integration_04.gauss_legendre_2d_integration import GaussLegendreDoubleIntegration

int_fun = lambda x, y: np.exp(-x ** 2 - y ** 2)  # 被积函数

gl2d = GaussLegendreDoubleIntegration(int_fun, [0, 1], [0, 1], zeros_num=10)
gl2d.cal_2d_int()
print(gl2d.int_value)
print("误差：", 0.557746285351034 - gl2d.int_value)