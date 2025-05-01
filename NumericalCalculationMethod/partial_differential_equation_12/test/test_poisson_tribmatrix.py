# -*- coding: UTF-8 -*-
"""
@file_name: test_poisson_equ.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from partial_differential_equation_12.pde_poisson_equation_matrix import PDEPoissonEquationTriBMatrix

# 例14代码
# g_xy = lambda x, y: 0.0 * x + 0.0 * y  # 方程右端定义
# f_u0y = lambda y: np.sin(y) + np.cos(y)
# f_uay = lambda y: np.exp(1) * (np.sin(y) + np.cos(y))
# f_ux0 = lambda x: np.exp(x)
# f_uxb = lambda x: np.exp(x) * (np.sin(1) + np.cos(1))
# x_span, y_span, n_x, n_y = [0, 1], [0, 1], 80, 80
# pde_model = lambda x, y: np.exp(x) * (np.sin(y) + np.cos(y))


# 例13代码
g_xy = lambda x, y: (np.pi ** 2 - 1) * np.exp(x) * np.sin(np.pi * y)  # 右端函数
f_u0y, f_uay = lambda y: np.sin(np.pi * y), lambda y: np.exp(2) * np.sin(np.pi * y)
f_ux0, f_uxb = lambda x: 0, lambda x: 0
x_span, y_span, n_x, n_y = [0, 2], [0, 1], 80, 80
pde_model = lambda x, y: np.exp(x) * np.sin(np.pi * y)

# 例19代码
# g_xy = lambda x, y: -6 * (x + y)
# f_u0y, f_uay = lambda y: y ** 3, lambda y: 1 + y ** 3
# f_ux0, f_uxb = lambda x: x ** 3, lambda x: 1 + x ** 3
# x_span, y_span, n_x, n_y = [0, 1], [0, 1], 80, 80
# pde_model = lambda x, y: x ** 3 + y ** 3

poisson = PDEPoissonEquationTriBMatrix(g_xy, f_ux0, f_uxb, f_u0y, f_uay, x_span, y_span, n_x, n_y,
                                       pde_model=pde_model, is_show=True)
poisson.solve_pde()
poisson.plt_pde_poisson_surface()
