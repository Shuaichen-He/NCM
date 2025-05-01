# -*- coding: UTF-8 -*-
"""
@file_name: test_pde_wave_equ.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from partial_differential_equation_12.pde_wave_equation import PDEWaveEquation

# 例3代码
f_fun = lambda x: np.sin(np.pi * x) + np.sin(2 * np.pi * x)  # 边界条件
g_fun = lambda x: 0 * x  # 一阶导数边界条件
pde_model = lambda x, t: np.sin(np.pi * x) * np.cos(2 * np.pi * t) + \
                         np.sin(2 * np.pi * x) * np.cos(4 * np.pi * t)  # 解析解
b_u0t_fun = [lambda t: 0 * t, lambda t: 0 * t]
x_a, t_b, c, x_h, t_h = 1, 0.5, 2, 0.0005, 0.0001  # 初始化参数

# 例4代码
# f_fun, g_fun = lambda x: np.exp(x), lambda x: np.exp(x)
# b_u0t_fun = [lambda t: np.exp(t), lambda t: np.exp(1 + t)]
# pde_model = lambda x,t: np.exp(x + t)  # 解析解
# x_a, t_b, c, x_h, t_h = 1, 1, 1, 0.0005, 5e-5  # 初始化参数

wave = PDEWaveEquation(f_fun, g_fun, b_u0t_fun, x_a, t_b, c, x_h, t_h, pde_model)  # 实例化对象
wave.cal_pde()  # 求解
wave.plt_pde_wave_surface()  # 可视化数值解及误差
wave.plt_pde_wave_curve_contourf()  # 可视化某些时刻的波的位移数值解和等值线图
