# -*- coding: UTF-8 -*-
"""
@file_name: exp3.py
@time: 2023-02-12
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from partial_differential_equation_12.pde_heat_conduction_2d_FRADI import PDEHeatConductionEquation_2D_FRADI
from partial_differential_equation_12.pde_heat_conduction_equ_2d import PDEHeatConductionEquation_2D


f_u0yt = lambda y, t: 0.0 * y + 0.0 * t  # u(0,y,t)
f_u1yt = lambda y, t: np.sin(y * t)  # u(1,y,t)
f_u0xt = lambda x, t: 0.0 * x + 0.0 * t  # u(x,0,t)
f_u1xt = lambda x, t: np.sin(x * t)  # u(x,1,t)
f_ut0 = lambda x, y: x * 0 + y * 0  # 初值条件
f_xyt = lambda x, y, t: (x ** 2 + y ** 2) * t ** 2 * np.sin(x * y * t) + x * y * np.cos(x * y * t)
pde_model = lambda x, y, t: np.sin(x * y * t)  # 解析解
a_const, x_span, y_span, t_T, tau, h = 1, [0, 1], [0, 1], 1, 0.0005, 0.02  # 修改步长

# Du Fort-Frankel 显式差分格式
heat_adi = PDEHeatConductionEquation_2D(a_const, f_xyt, f_u0yt, f_u1yt, f_u0xt, f_u1xt, f_ut0,
                                        x_span, y_span, t_T, h, tau, diff_type="du-fort-frankel",
                                        pde_model=pde_model)
u_xyt = heat_adi.solve_pde()
heat_adi.plt_pde_heat_surface()

# PR交替方向隐格式
head_adi = PDEHeatConductionEquation_2D_FRADI(a_const, f_xyt, f_ut0, f_u0yt, f_u1yt, f_u0xt, f_u1xt,
                                              x_span, y_span, t_T, tau, h, pde_model)
head_adi.solve_pde()
head_adi.plt_pde_heat_surface()
