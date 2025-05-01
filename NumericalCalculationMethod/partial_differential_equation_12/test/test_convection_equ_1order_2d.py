# -*- coding: UTF-8 -*-
"""
@file_name: test_convection_equ_1order_2d.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from partial_differential_equation_12.pde_convection_equation_1order_2d import PDEConvectionEquationFirstOrder2D_LF


f_u0 = lambda x, y: np.exp(-10 * x ** 2 - 10 * y ** 2)
a_const, b_const = 1, 1
x_span, y_span, t_T, x_n, y_n, t_m = [0, 1], [0, 1], 0.5, 100, 100, 100
cefo2 = PDEConvectionEquationFirstOrder2D_LF(a_const, b_const, f_u0, x_span, y_span, t_T, x_n, y_n, t_m)
cefo2.solve_pde()
# 可视化6个时刻的数值解曲面
fig = plt.figure(figsize=(16, 10))
ax_1 = fig.add_subplot(231, projection='3d')
cefo2.plt_convection_surf(0, is_show=False, ax=ax_1)
ax_2 = fig.add_subplot(232, projection='3d')
cefo2.plt_convection_surf(0.1, is_show=False, ax=ax_2)
ax_3 = fig.add_subplot(233, projection='3d')
cefo2.plt_convection_surf(0.2, is_show=False, ax=ax_3)
ax_4 = fig.add_subplot(234, projection='3d')
cefo2.plt_convection_surf(0.3, is_show=False, ax=ax_4)
ax_5 = fig.add_subplot(235, projection='3d')
cefo2.plt_convection_surf(0.4, is_show=False, ax=ax_5)
ax_6 = fig.add_subplot(236, projection='3d')
cefo2.plt_convection_surf(0.5, is_show=False, ax=ax_6)
plt.show()

