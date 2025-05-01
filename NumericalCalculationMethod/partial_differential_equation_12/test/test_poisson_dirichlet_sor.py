# -*- coding: UTF-8 -*-
"""
@file_name: test_poisson_dirichlet_sor.py
@time: 2021-12-05
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from partial_differential_equation_12.pde_poisson_equation_sor import PDEPoissonEquationSORIteration

# 例13代码
g_xy = lambda x, y: (np.pi ** 2 - 1) * np.exp(x) * np.sin(np.pi * y)  # 右端函数
f_u0y, f_uay = lambda y: np.sin(np.pi * y), lambda y: np.exp(2) * np.sin(np.pi * y)
f_ux0, f_uxb = lambda x: 0, lambda x: 0
x_a, y_b, x_h, y_h, eps = 2, 1, 0.01, 0.01, 1e-7
pde_model = lambda x, y: np.exp(x) * np.sin(np.pi * y)

# 例19代码
# g_xy = lambda x, y: -6 * (x + y)
# f_u0y, f_uay = lambda y: y ** 3, lambda y: 1 + y ** 3
# f_ux0, f_uxb = lambda x: x ** 3, lambda x: 1 + x ** 3
# x_a, y_b, x_h, y_h, eps = 1, 1, 0.05, 0.05, 1e-7
# pde_model = lambda x, y: x ** 3 + y ** 3


# g_xy = lambda x, y: 0
# left_b = lambda y: np.sin(y) + np.cos(y)
# right_b = lambda y: np.exp(1) * (np.sin(y) + np.cos(y))
# lower_b = lambda x: np.exp(x)
# upper_b = lambda x: np.exp(x) * (np.sin(1) + np.cos(1))
# pde_model = lambda x, y: np.exp(x) * (np.sin(y) + np.cos(y))
# pde_model = lambda x, y: np.exp(x) * np.sin(np.pi * y)


poisson = PDEPoissonEquationSORIteration(g_xy, f_ux0, f_uxb, f_u0y, f_uay, x_a, y_b, x_h, y_h, eps=eps,
                                      max_iter=1000, pde_model=pde_model)
poisson.solve_pde()
poisson.plt_pde_poisson_surface()
