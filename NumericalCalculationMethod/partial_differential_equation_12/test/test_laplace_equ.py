# -*- coding: UTF-8 -*-
"""
@file_name: test_laplace_equ.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from partial_differential_equation_12.pde_laplace_equation_dirichlet import PDELaplaceEquationDirichlet
import time

# f_ux0 = lambda x: 0
# f_uxb = lambda x: 180
# f_u0y = lambda y: 80
# f_uay = lambda y: 0
# x_a, y_b, x_h, y_h = 4, 4, 0.5, 0.5

f_ux0 = lambda x: x ** 4
f_uxb = lambda x: x ** 4 - 13.5 * x ** 2 + 5.0625
f_u0y = lambda y: y ** 4
f_uay = lambda y: y ** 4 - 13.5 * y ** 2 + 5.0625
x_a, y_b, x_h, y_h, eps = 1.5, 1.5, 0.01, 0.01, 1e-8  # 修改步长和精度
pde_model = lambda x, y: x ** 4 - 6 * x ** 2 * y ** 2 + y ** 4

laplace = PDELaplaceEquationDirichlet(f_ux0, f_uxb, f_u0y, f_uay, x_a, y_b, x_h, y_h, max_iter=1000,
                                      eps=eps, pde_model=pde_model)
time_start = time.time()
laplace.solve_pde()
time_end = time.time()
print("消耗时间：", time_end - time_start)
laplace.plt_pde_laplace_surface()