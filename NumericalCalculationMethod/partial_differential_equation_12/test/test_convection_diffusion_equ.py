# -*- coding: UTF-8 -*-
"""
@file_name: test_convection_diffusion_equ.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from partial_differential_equation_12.pde_convection_diffusion_equation import PDEConvectionDiffusionEquation

# 例8示例
f_ut0 = lambda x: np.exp(-x)
alpha_fun, beta_fun = lambda t: np.exp(1.01 * t), lambda t: np.exp(-1 + 1.01 * t)
a_const, b_const, x_span, t_T, x_h, t_h = 1, 0.01, [0, 1], 0.5, 0.01, 0.0001
pde_model = lambda x, t: np.exp(-x + 1.01 * t)

diff_scheme = ["center", "exp", "samarskii", "crank-nicolson"]
for scheme in diff_scheme:
    convect_diffusion = PDEConvectionDiffusionEquation(f_ut0, alpha_fun, beta_fun, a_const, b_const,
                                                       x_span, t_T, x_h, t_h, pde_model, diff_type=scheme)
    u_xt = convect_diffusion.solve_pde()
    convect_diffusion.plt_pde_heat_surface()  # 单个绘制

# # f_ut0 = lambda x: x ** 2
# # a_const, b_const, x_span, t_T, x_h, t_h = 1, 1, [0, 1], 0.5, 0.01, 0.00005
# xi = np.arange(0, 1 + 0.01, 0.01)
# ti = np.arange(0, 0.5 + 0.00005, 0.00005)
# x, t = np.meshgrid(xi, ti)
# fig = plt.figure(figsize=(14, 10))
# for i, scheme in enumerate(diff_scheme):
#     convect_diffusion = PDEConvectionDiffusionEquation(f_ut0, a_const, b_const,
#                                                        x_span, t_T, x_h, t_h, diff_type=scheme)
#     u_xt = convect_diffusion.solve_pde()
#     ax = fig.add_subplot(221 + i, projection='3d')
#     ax.plot_surface(x, t, u_xt.T, cmap='rainbow')
#     ax.set_xlabel("$x$", fontdict={"fontsize": 18})
#     ax.set_ylabel("$t$", fontdict={"fontsize": 18})
#     ax.set_zlabel("$U$", fontdict={"fontsize": 18})
#     plt.tick_params(labelsize=16)  # 刻度字体大小16
#     plt.title("对流扩散方程数值解曲面$(%s)$" % scheme, fontdict={"fontsize": 18})
#     # convect_diffusion.plt_pde_heat_surface()  # 单个绘制
#     # convect_diffusion.plt_pde_heat_curve_contourf()
# fig.tight_layout()
# plt.show()
