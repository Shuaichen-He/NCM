# -*- coding: UTF-8 -*-
"""
@file_name: test_gauss_2d_general_region.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_integration_04.gauss_2d_general_region import Gauss2DGeneralIntegration

fh = lambda x, y: np.exp(y / x)
a, b, c_x, d_x = 0.1, 0.5, lambda x: x ** 3, lambda x: x ** 2
g2dgi = Gauss2DGeneralIntegration(fh, a, b, c_x, d_x, np.array([10, 10]))
int_val = g2dgi.cal_2d_int()
print("积分近似值：", int_val, "误差：", 0.033305566116232 - int_val)

fh = lambda x, y: 3 * x ** 2 * y ** 2
a, b, c_x, d_x = 0, 1, lambda x: 0 * x, lambda x: 1 - x ** 2
g2dgi = Gauss2DGeneralIntegration(fh, a, b, c_x, d_x, np.array([5, 5]))
int_val = g2dgi.cal_2d_int()
print("积分近似值：", int_val, "误差：", 16 / 315 - int_val)
