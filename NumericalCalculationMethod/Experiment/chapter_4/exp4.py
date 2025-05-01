# -*- coding: UTF-8 -*-
"""
@file_name: exp4.py
@time: 2023-02-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_integration_04.gauss_2d_general_region import Gauss2DGeneralIntegration
from numerical_integration_04.gauss_3d_general_region import Gauss3DGeneralIntegration

# 一般区域的二重积分
fh = lambda x, y: 2 * y * np.sin(x) + np.cos(x) ** 2
a, b, c_x, d_x = 0, np.pi / 4, lambda x: np.sin(x), lambda x: np.cos(x)
I_f = (5 * np.sqrt(2) - 4) / 6
g2dgi = Gauss2DGeneralIntegration(fh, a, b, c_x, d_x, np.array([10, 10]))
int_val = g2dgi.cal_2d_int()
print("积分近似值：%.15f, 绝对值误差：%.15e" % (int_val, abs(I_f - int_val)))

# 一般区域的三重积分
fh = lambda x, y, z: np.sin(z / y) / y
I_f = 0.5 * np.pi ** 2 + 2
a, b, c_x, d_x = 0, np.pi, lambda x: 0 * x, lambda x: x
alpha_xy, beta_xy = lambda x, y: 0 * x * y, lambda x, y: x * y
g3dgi = Gauss3DGeneralIntegration(fh, a, b, c_x, d_x, alpha_xy, beta_xy, np.array([10, 10, 15]))
int_val = g3dgi.cal_3d_int()
print("积分近似值：%.15f, 绝对值误差：%.15e" % (int_val, abs(I_f - int_val)))