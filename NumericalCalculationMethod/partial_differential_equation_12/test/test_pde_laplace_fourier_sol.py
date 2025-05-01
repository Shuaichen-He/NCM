# -*- coding: UTF-8 -*-
"""
@file_name: test_pde_laplace_fourier_sol.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from partial_differential_equation_12.pde_laplace_fourier_sol import PDELaplaceFourierSolution

# 例16拉普拉斯方程
# u_y0 = lambda y: y ** 4
# u_ya = lambda y: y ** 4 - 13.5 * y ** 2 + 5.0625
# u_x0 = lambda x: x ** 4
# u_xb = lambda x: x ** 4 - 13.5 * x ** 2 + 5.0625
# a, b, m, n = 1.5, 1.5, 15, 15
# pde_model = lambda x, y: x ** 4 - 6 * x ** 2 * y ** 2 + y ** 4

u_y0 = lambda y: np.pi - y
u_ya = lambda y: 0.0 * y
u_x0 = lambda x: np.cos(x)
u_xb = lambda x: x * np.sin(x)
a, b, m, n = np.pi, np.pi, 15, 15
# pde_model = lambda x, y: x ** 4 - 6 * x ** 2 * y ** 2 + y ** 4

# 例17拉普拉斯方程
# u_y0 = lambda y: np.sin(y) + np.cos(y)
# u_ya = lambda y: np.exp(1) * (np.sin(y) + np.cos(y))
# u_x0 = lambda x: np.exp(x)
# u_xb = lambda x: np.exp(x) * (np.sin(1) + np.cos(1))
# a, b, m, n = 2, 2, 30, 30
# pde_model = lambda x, y: np.exp(x) * (np.sin(y) + np.cos(y))


laplace_fourier = PDELaplaceFourierSolution(u_x0, u_xb, u_y0, u_ya, a, b, m, n, pde_model=None)
laplace_fourier.solve_pde()
laplace_fourier.plt_pde_laplace_surface([0, np.pi], [0, np.pi])
# laplace_fourier.plt_pde_laplace_surface([0.3, 1], [0.3, 1])