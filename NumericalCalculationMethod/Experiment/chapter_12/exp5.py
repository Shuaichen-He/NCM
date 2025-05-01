# -*- coding: UTF-8 -*-
"""
@file_name: exp5.py
@time: 2023-02-12
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
from partial_differential_equation_12.pde_poisson_equation_matrix import PDEPoissonEquationTriBMatrix
from partial_differential_equation_12.pde_poisson_equation_sor import PDEPoissonEquationSORIteration

g_xy = lambda x, y: -6 * (x + y)
f_u0y, f_uay = lambda y: y ** 3, lambda y: 1 + y ** 3
f_ux0, f_uxb = lambda x: x ** 3, lambda x: 1 + x ** 3
pde_model = lambda x, y: x ** 3 + y ** 3

# 超松弛迭代法
x_a, y_b, x_h, y_h, eps = 1, 1, 0.05, 0.05, 1e-8
poisson = PDEPoissonEquationSORIteration(g_xy, f_ux0, f_uxb, f_u0y, f_uay, x_a, y_b, x_h, y_h, eps=eps,
                                      max_iter=1000, pde_model=pde_model)
poisson.solve_pde()
poisson.plt_pde_poisson_surface()

# 三对角块矩阵
x_span, y_span, n_x, n_y = [0, 1], [0, 1], 10, 10
poisson = PDEPoissonEquationTriBMatrix(g_xy, f_ux0, f_uxb, f_u0y, f_uay, x_span, y_span, n_x, n_y,
                                       pde_model=pde_model, is_show=True)
poisson.solve_pde()
poisson.plt_pde_poisson_surface()