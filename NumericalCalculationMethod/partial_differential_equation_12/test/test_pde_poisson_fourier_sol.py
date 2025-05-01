# -*- coding: UTF-8 -*-
"""
@file_name: test_pde_poisson_fourier_sol.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from partial_differential_equation_12.pde_poisson_fourier_sol import PDEPoissonFourierSolution

# 例18方程代码
# f_xy = lambda x, y:  -(np.pi ** 2 - 1) * np.exp(x) * np.sin(np.pi * y)  # 右端函数
# u_y0 = lambda y: np.sin(np.pi * y)
# u_ya = lambda y: np.exp(2) * np.sin(np.pi * y)
# u_x0 = lambda x: 0.0 * x
# u_xb = lambda x: 0.0 * x
# pde_model = lambda x, y: np.exp(x) * np.sin(np.pi * y)  # 解析解
# a, b, m, n = 2, 1, 10, 10

f_xy = lambda x, y: np.sin(x) * np.sin(y)  # 右端函数
pde_model = lambda x, y: -0.5 * np.sin(x) * np.sin(y)  # 解析解
a, b, m, n = 2 * np.pi, 2 * np.pi, 10, 10
laplace_fourier = PDEPoissonFourierSolution(f_xy, a, b, m, n, pde_model)
laplace_fourier.solve_pde()
laplace_fourier.plt_pde_poisson_surface([0, 2 * np.pi], [0, 2 * np.pi])

