# -*- coding: UTF-8 -*-
"""
@file_name: test_heat_conduction_equ.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from partial_differential_equation_12.pde_heat_conduction_equation import PDEHeatConductionEquation

f_fun = lambda x: np.sin(np.pi * x) + np.sin(3 * np.pi * x)  # 边界条件
# f_fun = lambda x: 4 * x - 4 * x ** 2  # 边界条件

# c1, c2, x_a, t_b, c, x_h, t_h = 0, 0, 1, 0.2, 1, 0.005, 0.00005  # 初始化参数，显示差分格式
# c1, c2, x_a, t_b, c, x_h, t_h = 0, 0, 1, 0.1, 1, 0.001, 0.0001  # 初始化参数，隐式差分格式

# c1, c2, x_a, t_b, c, x_h, t_h = 0, 0, 1, 0.1, 1, 0.01, 0.00001  # 初始化参数，显示差分格式
c1, c2, x_a, t_b, c, x_h, t_h = 0, 0, 1, 0.1, 1, 0.001, 0.0001  # 初始化参数，隐式差分格式
pde_model = lambda x, t: np.sin(np.pi * x) * np.exp(-np.pi ** 2 * t) + \
                         np.sin(3 * np.pi * x) * np.exp(-9 * np.pi ** 2 * t)

heat_c = PDEHeatConductionEquation(f_fun, c1, c2, x_a, t_b, c, x_h, t_h, pde_model, pde_method="implicit")  # explicit、implicit
heat_c.cal_pde()
heat_c.plt_pde_heat_surface()
heat_c.plt_pde_heat_curve_contourf()
