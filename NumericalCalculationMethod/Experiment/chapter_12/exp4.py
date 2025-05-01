# -*- coding: UTF-8 -*-
"""
@file_name: exp4.py
@time: 2023-02-12
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from partial_differential_equation_12.pde_laplace_equation_dirichlet import PDELaplaceEquationDirichlet
import time

f_u0y = lambda y: np.sin(y) + np.cos(y)  # u(0,y)
f_uay = lambda y: np.exp(1) * (np.sin(y) + np.cos(y))  # u(1,y)
f_ux0 = lambda x: np.exp(x) # u(x,0)
f_uxb = lambda x: np.exp(x) * (np.sin(1) + np.cos(1)) # u(x,1)
pde_model = lambda x, y: np.exp(x) * (np.sin(y) + np.cos(y))  # 解析解
x_a, y_b, x_h, y_h, eps = 1, 1, 0.01, 0.01, 1e-8

laplace = PDELaplaceEquationDirichlet(f_ux0, f_uxb, f_u0y, f_uay, x_a, y_b, x_h, y_h, max_iter=1000,
                                      eps=eps, pde_model=pde_model)
time_start = time.time()
laplace.solve_pde()
time_end = time.time()
print("消耗时间：", time_end - time_start)
laplace.plt_pde_laplace_surface()
