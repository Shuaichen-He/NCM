# -*- coding: UTF-8 -*-
"""
@file_name: exp1.py
@time: 2023-02-12
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from partial_differential_equation_12.pde_wave_equation_mixed_boundary import PDEWaveEquationMixedBoundary
from partial_differential_equation_12.pde_wave_equation_mixed_boundary_implicit \
    import PDEWaveEquationMixedBoundaryImplicit

fun_xt = lambda t, x: 0.0 * t + 0.0 * x  # 右端方程
alpha_fun, beta_fun = lambda t: np.exp(t), lambda t: np.exp(1 + t)
u_x0, du_x0 = lambda x: np.exp(x), lambda x: np.exp(x)
pde_model = lambda t, x: np.exp(x + t)  # 解析解

x_a, t_T, c, x_h, t_h = 1, 1, 1, 0.0005, 0.0005

# 显式格式
wave = PDEWaveEquationMixedBoundary(fun_xt, alpha_fun, beta_fun, u_x0, du_x0,
                                    x_a, t_T, c, x_h, t_h, pde_model=pde_model)
wave.solve_pde()
wave.plt_pde_wave_surface()
wave.plt_pde_wave_curve_contourf()

# 隐式格式
wave = PDEWaveEquationMixedBoundaryImplicit(fun_xt, alpha_fun, beta_fun, u_x0, du_x0,
                                            x_a, t_T, c, x_h, t_h, pde_model=pde_model)
wave.solve_pde()
wave.plt_pde_wave_surface()
wave.plt_pde_wave_curve_contourf()