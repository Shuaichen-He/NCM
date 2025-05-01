# -*- coding: UTF-8 -*-
"""
@file_name: exp2.py
@time: 2023-02-12
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from partial_differential_equation_12.pde_heat_conduction_equ_nonhomogeneous \
    import PDEHeatConductionEquationNonhomogeneous

f_fun = lambda x, t: -np.exp(x) * (np.cos(0.5 - t) + 2 * np.sin(0.5 - t))
init_f = lambda x: np.exp(x) * np.sin(0.5)
alpha_f, beta_f = lambda t: np.sin(0.5 - t), lambda t: np.exp(1) * np.sin(0.5 - t)
a, x_a, t_T, x_h, t_h = 2, 1, 1, 0.01, 0.00001
pde_model = lambda x, t: np.exp(x) * np.sin(0.5 - t)

diff_scheme = ["forward", "backward", "crank-nicolson", "compact"]
for scheme in diff_scheme:
    print("方法：", scheme)
    heat = PDEHeatConductionEquationNonhomogeneous(f_fun, a, init_f, alpha_f, beta_f,
                                                   x_a, t_T, x_h, t_h, pde_model, diff_type=scheme)
    heat.solve_pde()
    heat.plt_pde_heat_surface()
    heat.plt_pde_heat_curve_contourf()
    print("=" * 60)
