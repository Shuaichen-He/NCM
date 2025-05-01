# -*- coding: UTF-8 -*-
"""
@file_name: test_wave_mixed.py
@time: 2021-12-06
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from partial_differential_equation_12.pde_wave_equation_mixed_boundary import PDEWaveEquationMixedBoundary

# 例4示例
fun_xt = lambda t, x: (t ** 2 - x ** 2) * np.sin(x * t)  # 右端方程
alpha_fun, beta_fun = lambda t: 0.0 * t, lambda t: np.sin(t)
u_x0, du_x0 = lambda x: 0.0 * x, lambda x: x
pde_model = lambda t, x: np.sin(x * t)  # 解析解
wave = PDEWaveEquationMixedBoundary(fun_xt, alpha_fun, beta_fun, u_x0, du_x0,
                                    1, 1, 1, 0.001, 0.0005, pde_model=pde_model)
wave.solve_pde()
wave.plt_pde_wave_surface()
wave.plt_pde_wave_curve_contourf()
