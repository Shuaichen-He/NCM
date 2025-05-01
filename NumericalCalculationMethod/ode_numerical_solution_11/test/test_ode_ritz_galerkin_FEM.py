# -*- coding: UTF-8 -*-
"""
@file_name: test_ode_ritz_galerkin_FEM.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from ode_numerical_solution_11.ode_ritz_galerkin_FEM import ODERitzGalerkinFEM

x, k = sympy.symbols("x, k")  # 定义基函数的符号变量
f_ux = (1 + np.pi ** 2) * sympy.sin(np.pi * x)
x_span, n = np.array([0, 1]), 60  # 此处修改子空间数n以及求解区间
ode_model = lambda x: np.sin(np.pi * x)  # 解析解

ode_rg = ODERitzGalerkinFEM(f_ux, x_span, n, ode_model)
ode_rg.fit_ode()
ode_rg.plt_ode_curve()
