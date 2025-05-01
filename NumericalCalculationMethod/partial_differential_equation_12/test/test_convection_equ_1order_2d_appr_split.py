# -*- coding: UTF-8 -*-
"""
@file_name: test_convection_equ_1order_2d_appr_split.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from partial_differential_equation_12.pde_convection_equ_1order_2d_appr_split \
    import PDEConvectionEquationFirstOrder2D_ApprSplit


f_u0 = lambda x, y: np.exp(-10 * x ** 2 - 10 * y ** 2)
a_const, b_const = 1, 1
x_span, y_span, x_n, y_n, t_m = [0, 1], [0, 1], 100, 100, 100
fig = plt.figure(figsize=(14, 5))
t_T = 0.25
convection = PDEConvectionEquationFirstOrder2D_ApprSplit(a_const, b_const, f_u0,
                                                  x_span, y_span, t_T, x_n, y_n, t_m)
convection.solve_pde()
ax_1 = fig.add_subplot(121, projection='3d')
convection.plt_convection_surf(is_show=False, ax=ax_1)
t_T = 0.5
convection = PDEConvectionEquationFirstOrder2D_ApprSplit(a_const, b_const, f_u0,
                                                  x_span, y_span, t_T, x_n, y_n, t_m)
convection.solve_pde()
ax_2 = fig.add_subplot(122, projection='3d')
convection.plt_convection_surf(is_show=False, ax=ax_2)
plt.show()
