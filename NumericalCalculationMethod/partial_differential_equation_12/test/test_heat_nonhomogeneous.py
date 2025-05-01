# -*- coding: UTF-8 -*-
"""
@file_name: test_heat_nonhomogeneous.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from partial_differential_equation_12.pde_heat_conduction_equ_nonhomogeneous \
    import PDEHeatConductionEquationNonhomogeneous

# 例7方程代码
f_fun, init_f = lambda x, t: 0.0 * x + 0.0 * t, lambda x: np.exp(x)
alpha_f, beta_f = lambda t: np.exp(t), lambda t: np.exp(1 + t)
a, x_a, t_T, x_h, t_h = 1, 1, 1, 0.01, 0.00001
pde_model = lambda x, t: np.exp(x + t)

diff_scheme = ["forward", "backward", "crank-nicolson", "compact"]
for scheme in diff_scheme:
    heat = PDEHeatConductionEquationNonhomogeneous(f_fun, a, init_f, alpha_f, beta_f,
                                               x_a, t_T, x_h, t_h, pde_model, diff_type=scheme)
    heat.solve_pde()
    heat.plt_pde_heat_surface()
    heat.plt_pde_heat_curve_contourf()