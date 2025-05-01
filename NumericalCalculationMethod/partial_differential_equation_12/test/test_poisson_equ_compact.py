# -*- coding: UTF-8 -*-
"""
@file_name: test_poisson_equ_compact.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from partial_differential_equation_12.pde_poisson_equation_compact import PDEPoissonEquationCompact

# f_fun = lambda x, y: (np.pi ** 2 - 1) * np.exp(x) * np.sin(np.pi * y)  # 右端函数
#
# left_b = lambda y: np.sin(np.pi * y)
# right_b = lambda y: np.exp(2) * np.sin(np.pi * y)
# lower_b = lambda x: np.zeros(len(x))
# upper_b = lambda x: np.zeros(len(x))
#
# pde_model = lambda x, y: np.exp(x) * np.sin(np.pi * y)  # 解析解

g_xy = lambda x, y: 0.0 * x + 0.0 * y  # 方程右端定义
left_b = lambda y: np.sin(y) + np.cos(y)
right_b = lambda y: np.exp(1) * (np.sin(y) + np.cos(y))
lower_b = lambda x: np.exp(x)
upper_b = lambda x: np.exp(x) * (np.sin(1) + np.cos(1))
pde_model = lambda x, y: np.exp(x) * (np.sin(y) + np.cos(y))

poisson = PDEPoissonEquationCompact(g_xy, left_b, right_b, lower_b, upper_b, [0, 1], [0, 1], 30, 30,
                             pde_model=pde_model, is_show=True)
poisson.solve_pde()
poisson.plt_pde_poisson_surface()
poisson.plt_pde_poisson_curve_contourf()