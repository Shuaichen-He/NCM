# -*- coding: UTF-8 -*-
"""
@file_name: test_heat_conduction_equ_2d.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from partial_differential_equation_12.pde_heat_conduction_equ_2d import PDEHeatConductionEquation_2D

# f_xyt = lambda x, y, t: -1.5 * np.exp(0.5 * (x + y) - t)
# f_ux0, f_ux = lambda y, t: np.exp(0.5 * y - t), lambda y, t: np.exp(0.5 * (1 + y) - t)
# f_uy0, f_uy = lambda x, t: np.exp(0.5 * x - t), lambda x, t: np.exp(0.5 * (1 + x) - t)
# f_ut0 = lambda x, y: np.exp(0.5 * (x + y))
# a_const, x_span, y_span, t_T, xy_h, t_h = 1, [0, 2], [0, 2], 1, 0.05, 0.0005  # 求解参数
# pde_model = lambda x, y, t: np.exp(0.5 * (x + y) - t)  # 解析解

# f_xyt = lambda x, y, t: 0.0 * x + 0.0 * y + 0.0 * t
# f_ux0, f_ux = lambda y, t: 0.0 * y + 0.0 * t, lambda y, t: 0.0 * y + 0.0 * t
# f_uy0, f_uy = lambda x, t: 0.0 * x + 0.0 * t, lambda x, t: 0.0 * x + 0.0 * t
# f_ut0 = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
# a_const, x_span, y_span, t_T, xy_h, t_h = 1/8, [0, 2], [0, 2], 1, 0.1, 0.001  # 求解参数
# pde_model = lambda x, y, t: np.exp(-np.pi ** 2 * t / 8) * np.sin(np.pi * x) * np.sin(np.pi * y)  # 解析解

# f_xyt = lambda x, y, t: 0.0 * x + 0.0 * y + 0.0 * t
# f_ux0, f_ux = lambda y, t: 0.0 * y + 0.0 * t, lambda y, t: 0.0 * y + 0.0 * t
# f_uy0, f_uy = lambda x, t: 0.0 * x + 0.0 * t, lambda x, t: 0.0 * x + 0.0 * t
# f_ut0 = lambda x, y: np.sin(4 * np.pi * x) + np.cos(4 * np.pi * y)
# a_const, x_span, y_span, t_T, xy_h, t_h = 1, [0, 1], [0, 1], 0.1, 0.01, 0.0001  # 求解参数

# f_xyt = lambda x, y, t: 0.0 * x + 0.0 * y + 0.0 * t
# f_ux0, f_ux = lambda y, t: 0.0 * y + 0.0 * t, lambda y, t: 0.0 * y + 0.0 * t
# f_uy0, f_uy = lambda x, t: 0.0 * x + 0.0 * t, lambda x, t: 0.0 * x + 0.0 * t
# # f_ut0 = lambda x, y: 10 * np.sin(x) * np.sin(y)
# f_ut0 = lambda x, y: np.exp(-10 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
# a_const, x_span, y_span, t_T, xy_h, t_h = 1, [0, 2], [0, 2], 0.1, 0.01, 0.000005  # 求解参数
# pde_model = lambda x, y, t: 10 * np.sin(x) * np.sin(y) * np.exp(-2 * t)


# f_xyt = lambda x, y, t: 0.0 * x + 0.0 * y + 0.0 * t
# f_ux0, f_ux = lambda y, t: 0.0 * y + 0.0 * t, lambda y, t: 0.0 * y + 0.0 * t
# f_uy0, f_uy = lambda x, t: 0.0 * x + 0.0 * t, lambda x, t: 0.0 * x + 0.0 * t
# f_ut0 = lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y)
# a_const, x_span, y_span, t_T, xy_h, t_h = 0.5, [0, 1], [0, 1], 0.5, 0.01, 0.00005  # 求解参数
# pde_model = lambda x, y, t: np.sin(np.pi * x) * np.cos(np.pi * y) * np.exp(-np.pi ** 2 * t)

# 边界条件
f_u0yt = lambda y, t: np.exp(0.5 * y - t)
f_u1yt = lambda y, t: np.exp(0.5 * (1 + y) - t)
f_u0xt = lambda x, t: np.exp(0.5 * x - t)
f_u1xt = lambda x, t: np.exp(0.5 * (1 + x) - t)
f_ut0 = lambda x, y: np.exp(0.5 * (x + y))  # 初值条件
f_xyt = lambda x, y, t: -1.5 * np.exp(0.5 * (x + y) - t)
pde_model = lambda x, y, t: np.exp(0.5 * (x + y) - t)  # 解析解
a_const, x_span, y_span, t_T, xy_h, t_h = 1, [0, 1], [0, 1], 1, 0.01, 0.00005  # 求解参数

# f_u0yt, f_u1yt = lambda y, t: 0.0 * y + 0.0 * t, lambda y, t: 0.0 * y + 0.0 * t
# f_u0xt = lambda x, t: np.sin(np.pi * x) * np.exp(-np.pi ** 2 * t)
# f_u1xt = lambda x, t: -np.sin(np.pi * x) * np.exp(-np.pi ** 2 * t)
# f_ut0 = lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y)  # 初始条件
# f_xyt = lambda x, y, t: 0.0 * x + 0.0 * y + 0.0 * t
# pde_model = lambda x, y, t: (x * np.sin(np.pi * y) - y * np.sin(np.pi * x)) * np.exp(-np.pi ** 2 * t)
# a_const, x_span, y_span, t_T, xy_h, t_h = 1, [0, 1], [0, 1], 0.2, 0.005, 0.00001  # 求解参数


heat_adi = PDEHeatConductionEquation_2D(a_const, f_xyt, f_u0yt, f_u1yt, f_u0xt, f_u1xt, f_ut0,
                                        x_span, y_span, t_T, xy_h, t_h, diff_type="du-fort-frankel",
                                        pde_model=pde_model)
u_xyt = heat_adi.solve_pde()
heat_adi.plt_pde_heat_surface()
# x, y = np.meshgrid(xi, yi)
# fig = plt.figure(figsize=(18, 12))
# print(u_xyt.shape)
# for i in range(9):
#     ax = fig.add_subplot(331 + i, projection='3d')
#     ax.plot_surface(x, y, u_xyt[:, :, 1000 * i].T, cmap='rainbow')
#     plt.title("Du Fort-Frankel t = %.3f" % (t_h * 1000 * i))
# plt.show()
#
#
# x_, y_, t_ = np.meshgrid(xi, yi, ti)
# pde_sol = pde_model(x_, y_, t_)
# print(x.shape, pde_sol[:,:, -1].T.shape)
# fig = plt.figure(figsize=(18, 12))
# for i in range(9):
#     ax = fig.add_subplot(331 + i, projection='3d')
#     ax.plot_surface(x, y, pde_sol[:, :, 1000 * i].T, cmap='rainbow')
#     plt.title("Du Fort-Frankel t = %.3f" % (t_h * 1000 * i))
# plt.show()
