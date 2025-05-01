# -*- coding: UTF-8 -*-
"""
@file_name: test_ode_ritz_galerkin.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from ode_numerical_solution_11.ode_ritz_galerkin import ODERitzGalerkin

x, k = sympy.symbols("x, k")  # 定义基函数的符号变量
f_ux = (1 + np.pi ** 2) * sympy.sin(np.pi * x)
x_span, n = np.array([0, 1]), 10
# phi_0 = 0

# f_ux = x ** 2 - x
# x_span, n = np.array([0, 1]), 10
# phi_0 = x

# f_ux = 2 * sympy.cos(x) + 2 / np.pi * x - 1
# x_span, n = np.array([0, np.pi]), 12
# phi_0 = 0

basis_func = "sin"
ode_model = lambda x: np.sin(np.pi * x)  # 解析解
# ode_model = lambda x: x ** 2 - (2 * np.exp(x)) / (np.exp(1) + 1) - (2 * np.exp(-x) * np.exp(1)) / (np.exp(1) + 1) + 2
# ode_model = lambda x: np.cos(x) + 2 / np.pi * x - 1

ode_rg = ODERitzGalerkin(f_ux, x_span, n, basis_func, ode_model)
ode_rg.fit_ode()
ode_rg.plt_ode_curve()
