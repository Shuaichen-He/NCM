# -*- coding: UTF-8 -*-
"""
@file_name: test_ode_rayleigh-ritz_FEM.py
@time: 2023-02-17
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
from ode_numerical_solution_11.ode_rayleigh_ritz_FEM import ODERayleighRitzFEM

x = sympy.symbols("x")
# f_ux, p_x, q_x = 2 * sympy.pi ** 2 * sympy.sin(sympy.pi * x), 1, sympy.pi ** 2
# f_ux, p_x, q_x = (1 + np.pi ** 2) * sympy.sin(np.pi * x), 1, 1
# f_ux, p_x, q_x = 4 * x ** 2 - 8 * x + 1, x, 4
f_ux, p_x, q_x = sympy.pi ** 2 / 16 * sympy.cos(sympy.pi / 4 * x), -1, sympy.pi ** 2 / 4
# ode_model = lambda x: np.sin(np.pi * x)
# ode_model = lambda x: np.sin(np.pi * x)  # 解析解
# ode_model = lambda x: x ** 2 - x
ode_model = lambda x: -1 / 3 * np.cos(np.pi / 2 * x) - np.sqrt(2) / 6 * np.sin(np.pi / 2 * x) + \
                      1 / 3 * np.cos(np.pi / 4 * x)
x_span, n = [0, 1], 6
ode_rr = ODERayleighRitzFEM(f_ux, p_x, q_x, x_span, n, ode_model)
ux = ode_rr.fit_ode()
print("数值解：\n", ux)
xi = np.linspace(0, 1, n + 2)
print("解析解：\n", ode_model(xi))
ode_rr.plt_ode_curve()
print("误差绝对值：\n", np.abs(ux - ode_model(xi)))