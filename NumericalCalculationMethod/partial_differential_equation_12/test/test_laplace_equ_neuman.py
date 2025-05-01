# -*- coding: UTF-8 -*-
"""
@file_name: test_laplace_equ.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from partial_differential_equation_12.pde_laplace_equation_neumann import PDELaplaceEquationNeumann
import time

# df_ux0 = lambda x: 0
# f_uxb = lambda x: 180
# f_u0y = lambda y: 80
# f_uay = lambda y: 0
# x_a, y_b, x_h, y_h = 4, 4, 0.5, 0.5

f_ux0 = lambda x: x ** 4
df_ux0 = lambda x: 4 * x ** 3
f_uxb = lambda x: x ** 4 - 13.5 * x ** 2 + 5.0625
df_u0y = lambda y: 4 * y ** 3
f_uay = lambda y: y ** 4 - 13.5 * y ** 2 + 5.0625
x_a, y_b, x_h, y_h = 1.5, 1.5, 0.05, 0.05

# f_ux0 = lambda x: 0
# f_uxb = lambda x: x * (x < 1) + (2 - x) * (x > 1)
# f_u0y = lambda y: 0
# f_uay = lambda y: y * (2 - y)
# x_a, y_b, x_h, y_h = 2, 2, 0.05, 0.05

pde_model = lambda x, y: x ** 4 - 6 * x ** 2 * y ** 2 + y ** 4

laplace = PDELaplaceEquationNeumann(x_a, y_b, x_h, y_h, eps=1e-3, max_iter=500, pde_model=pde_model,
                                    df_ux0=df_ux0, f_uxb=f_uxb, df_u0y=df_u0y, f_uay=f_uay)
time_start = time.time()
laplace.solve_pde()
time_end = time.time()
print("消耗时间：", time_end - time_start)
laplace.plt_pde_laplace_surface()
laplace.plt_pde_laplace_curve_contourf()