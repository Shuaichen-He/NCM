# -*- coding: UTF-8 -*-
"""
@file_name: test_pde_2d_heat_fourier_sol.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from partial_differential_equation_12.pde_2d_heat_fourier_sol import PDE2DHeatFourierSolution

x, y = sympy.symbols("x, y")
# f_xyt_0 = 100
f_xyt_0 = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)  # 初值条件
c, a, b, t_T = 1 / 4, 1, 1, 0.5
m, n = 5, 5
pde_model = lambda x, y, t: np.exp(-np.pi ** 2 / 8 * t) * np.sin(np.pi * x) * np.sin(np.pi * y)  # 解析解

heat_fourier = PDE2DHeatFourierSolution(f_xyt_0, c, a, b, t_T, m, n, pde_model)
heat_fourier.solve_pde()
heat_fourier.plt_pde_heat_surface()
