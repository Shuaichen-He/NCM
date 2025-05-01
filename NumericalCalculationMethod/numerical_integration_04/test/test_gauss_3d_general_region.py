# -*- coding: UTF-8 -*-
"""
@file_name: test_gauss_3d_general_region.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_integration_04.gauss_3d_general_region import Gauss3DGeneralIntegration

fh = lambda x, y, z: np.exp(x ** 2 + y ** 2 + 0 * z)
a, b, c_x, d_x, alpha_xy, beta_xy = 0, 1, lambda x: 0 * x, lambda x: 0 * x + 1, lambda x, y: - x * y, lambda x, y: x * y
g3dgi = Gauss3DGeneralIntegration(fh, a, b, c_x, d_x, alpha_xy, beta_xy, np.array([10, 10, 10]))
int_val = g3dgi.cal_3d_int()
print("积分近似值：", int_val, "误差：", (np.exp(1) - 1) ** 2 / 2 - int_val)

fh = lambda x, y, z: np.sin(z / y) / y + 0 * x
a, b, c_x, d_x, alpha_xy, beta_xy = 0, np.pi, lambda x: 0 * x, lambda x: x, lambda x, y: 0 * x * y, lambda x, y: x * y
g3dgi = Gauss3DGeneralIntegration(fh, a, b, c_x, d_x, alpha_xy, beta_xy, np.array([15, 15, 15]))
int_val = g3dgi.cal_3d_int()
print("积分近似值：", int_val, "误差：", np.pi ** 2 / 2 + 2 - int_val)
