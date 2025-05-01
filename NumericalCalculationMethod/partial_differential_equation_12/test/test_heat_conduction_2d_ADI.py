# -*- coding: UTF-8 -*-
"""
@file_name: test_heat_conduction_2d_ADI.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from partial_differential_equation_12.pde_heat_conduction_2d_ADI import PDEHeatConductionEquation_2D_ADI

# f_u0yt, f_u1yt = lambda y, t: 0.0 * y + 0.0 * t, lambda y, t: 0.0 * y + 0.0 * t
# f_u0xt = lambda x, t: np.sin(np.pi * x) * np.exp(-np.pi ** 2 * t)
# f_u1xt = lambda x, t: -np.sin(np.pi * x) * np.exp(-np.pi ** 2 * t)
# f_ut0 = lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y)  # 初始条件
# a_const, x_span, y_span, t_T, x_n, y_n, t_m = 0.5, [0, 1], [0, 1], 0.15, 200, 200, 100  # 求解参数
# pde_model = lambda x, y, t: np.sin(np.pi * x) * np.cos(np.pi * y) * np.exp(-np.pi ** 2 * t)

# f_u0yt = lambda y, t: 0.0 * y + 0.0 * t
# f_u1yt = lambda y, t: np.sin(np.pi * y) * np.exp(-np.pi ** 2 * t)
# f_u0xt = lambda x, t: 0.0 * x + 0.0 * t
# f_u1xt = lambda x, t: -np.sin(np.pi * x) * np.exp(-np.pi ** 2 * t)
# f_ut0 = lambda x, y: x * np.sin(np.pi * y) - y * np.sin(np.pi * x)  # 初始条件
# f_xyt = lambda x, y, t: 0.0 * y + 0.0 * t + 0.0 * x
# a_const, x_span, y_span, t_T, x_n, y_n, t_m = 1, [0, 1], [0, 1], 0.2, 200, 200, 500  # 求解参数
# pde_model = lambda x, y, t: (x * np.sin(np.pi * y) - y * np.sin(np.pi * x)) * np.exp(-np.pi ** 2 * t)

# # 边界条件
f_u0yt = lambda y, t: 0.0 * y + 0.0 * t
f_u1yt = lambda y, t: 10 * np.sin(1) * np.sin(y) * np.exp(-2 * t)
f_u0xt = lambda x, t: 0.0 * x + 0.0 * t
f_u1xt = lambda x, t: 10 * np.sin(x) * np.sin(1) * np.exp(-2 * t)
f_ut0 = lambda x, y: 10 * np.sin(x) * np.sin(y)  # 初值条件
f_xyt = lambda x, y, t: 0.0 * y + 0.0 * t + 0.0 * x
pde_model = lambda x, y, t: 10 * np.sin(x) * np.sin(y) * np.exp(-2 * t)  # 解析解
a_const, x_span, y_span, t_T, x_n, y_n, t_m = 1, [0, 1], [0, 1], 0.5, 100, 100, 100

# 边界条件
# f_u0yt = lambda y, t: np.exp(0.5 * y - t)
# f_u1yt = lambda y, t: np.exp(0.5 * (1 + y) - t)
# f_u0xt = lambda x, t: np.exp(0.5 * x - t)
# f_u1xt = lambda x, t: np.exp(0.5 * (1 + x) - t)
# f_ut0 = lambda x, y: np.exp(0.5 * (x + y))  # 初值条件
# f_xyt = lambda x, y, t: -1.5 * np.exp(0.5 * (x + y) - t)
# pde_model = lambda x, y, t: np.exp(0.5 * (x + y) - t)  # 解析解
# a_const, x_span, y_span, t_T, x_n, y_n, t_m = 1, [0, 1], [0, 1], 1, 100, 100, 500

# f_u0yt = lambda y, t: 0.0 * y + 0.0 * t
# f_u1yt = lambda y, t: 0.0 * y + 0.0 * t
# f_u0xt = lambda x, t: 0.0 * x + 0.0 * t
# f_u1xt = lambda x, t: 0.0 * x + 0.0 * t
# f_ut0 = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)  # 初值条件
# f_xyt = lambda x, y, t: 0.0 * y + 0.0 * t + 0.0 * x
# pde_model = lambda x, y, t: np.exp(-np.pi ** 2 / 8 * t) * np.sin(np.pi * x) * np.sin(np.pi * y) # 解析解
# a_const, x_span, y_span, t_T, x_n, y_n, t_m = 1 / 16, [0, 1], [0, 1], 0.5, 100, 100, 200

head_adi = PDEHeatConductionEquation_2D_ADI(a_const, f_xyt, f_ut0, f_u0yt, f_u1yt, f_u0xt, f_u1xt,
                                            x_span, y_span, t_T, x_n, y_n, t_m, pde_model)
head_adi.solve_pde()
head_adi.plt_pde_heat_surface()

xi, yi = np.linspace(0, 1, x_n + 1), np.linspace(0, 1, y_n + 1)
x, y = np.meshgrid(xi, yi)
fig = plt.figure(figsize=(14, 12))
for i in range(9):
    t_T = 0.1 + i * 0.1
    t_m = int(t_T / 0.002)
    head_adi = PDEHeatConductionEquation_2D_ADI(a_const, f_xyt, f_ut0, f_u0yt, f_u1yt, f_u0xt, f_u1xt,
                                                x_span, y_span, t_T, x_n, y_n, t_m, pde_model)
    u_xyt = head_adi.solve_pde()
    ax = fig.add_subplot(331 + i, projection='3d')
    ax.plot_surface(x, y, u_xyt, cmap='rainbow')
    ax.set_zlabel("$U(x,y,t), t = %.3f$" % (t_T))
fig.tight_layout()
plt.show()
