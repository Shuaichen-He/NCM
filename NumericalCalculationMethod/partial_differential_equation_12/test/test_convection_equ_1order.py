# -*- coding: UTF-8 -*-
"""
@file_name: test_convection_equ_1order.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from partial_differential_equation_12.pde_convection_equation_1order_1d import PDEConvectionEquationFirstOrder1D

# 例1
f_uxt = lambda x, t: np.exp(-200 * (x - 0.25 - t) ** 2)  # 解析解
f_u0 = lambda x: np.exp(-200 * (x - 0.25) ** 2)  # 初值函数
a_const = 1  # 常系数
x_span, t_T, x_j, t_n = [0, 1], 0.5, 200, 800
plt.figure(figsize=(14, 12))
diff_type = ["", "upwind", "leapfrog", "lax-wendroff", "lax-friedrichs", "beam-warming"]
xi = np.linspace(0, 1, 100)
for i, type in enumerate(diff_type):
    convection = PDEConvectionEquationFirstOrder1D(a_const, f_u0, x_span, t_T, x_j, t_n, diff_type=type)
    convection.solve_pde()
    plt.subplot(321 + i)
    ti = convection.plt_convection_curve(is_show=False)
    for k in range(5):
        plt.plot(xi, f_uxt(xi, ti[k]), "k--")  # 绘制对应时刻的解析解
plt.show()

fig = plt.figure(figsize=(14, 5))
xi, ti = np.linspace(0, 1, x_j + 1), np.linspace(0, 0.5, t_n + 1)
x, t = np.meshgrid(xi, ti)
ax = fig.add_subplot(121, projection='3d')
convection = PDEConvectionEquationFirstOrder1D(a_const, f_u0, x_span, t_T, x_j, t_n, diff_type="lax-wendroff")
u_xyt = convection.solve_pde()
ax.plot_surface(x, t, u_xyt, cmap='rainbow')
ax.set_xlabel("$x$", fontdict={"fontsize": 18})
ax.set_ylabel("$t$", fontdict={"fontsize": 18})
ax.set_zlabel("$U$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.title("对流方程的数值解曲面($lax-wendroff$)", fontdict={"fontsize": 18})
ax = fig.add_subplot(122, projection='3d')
err = f_uxt(x, t) - u_xyt
print("最大绝对误差：%.10e" % np.max(np.abs(err)))
ax.plot_surface(x, t, err, cmap='rainbow')
ax.set_xlabel("$x$", fontdict={"fontsize": 18})
ax.set_ylabel("$t$", fontdict={"fontsize": 18})
ax.set_zlabel("$\epsilon$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.title("误差曲面$\epsilon=U(x,t) - \hat U(x,t),\ MAE=%.3e$" % np.mean(np.abs(err)), fontdict={"fontsize": 18})
plt.show()