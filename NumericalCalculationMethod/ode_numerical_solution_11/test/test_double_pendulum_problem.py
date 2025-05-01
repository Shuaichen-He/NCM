# -*- coding: UTF-8 -*-
"""
@file_name: test_double_pendulum_problem.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from ode_numerical_solution_11.first_order_ODEs_RK import FirstOrderODEsRK


# 定义微分方程组
def ode_pendulum(t, y):
    s_v, c_v = np.sin(y[0] - y[2]), np.cos(y[0] - y[2])
    dy2 = (y[3] ** 2 * s_v + 58.8 * np.sin(y[0]) + 2 * y[1] ** 2 * s_v * c_v - 9.8 * c_v * np.sin(y[2])) \
          / (2 * c_v ** 2 - 12)
    dy4 = (12 * y[1] ** 2 * s_v - 58.8 * np.sin(y[2]) + y[3] ** 2 * s_v * c_v + 58.8 * c_v * np.sin(y[0])) / \
          (6 - c_v ** 2)
    return np.array([y[1], dy2, y[3], dy4])


def ax_split():
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=3)
    ax2 = plt.subplot2grid((2, 5), (1, 0), colspan=3)
    ax3 = plt.subplot2grid((2, 5), (0, 3), colspan=2, rowspan=2)
    return ax1, ax2, ax3


x0, y0, h = 0, np.array([2.0, 0.0, 0, 0.0]), 0.001  # 初始值及求解微分步长
odes_rk = FirstOrderODEsRK(ode_pendulum, x0, y0, x_final=20, h=h)  # 实例化
odes_rk.fit_odes()  # 求解微分方程组

# 可视化，三个子图
ax1, ax2, ax3 = ax_split()
# 第一摆x1
ax1.plot(odes_rk.ode_sol[:, 0], odes_rk.ode_sol[:, 1], 'r')
ax1.set_ylabel(r'$\theta_1$', fontsize=18)
ax1.tick_params(labelsize=16)  # 刻度字体大小16
# 第二摆x2
ax2.plot(odes_rk.ode_sol[:, 0], odes_rk.ode_sol[:, 3], 'g')
ax2.set_xlabel(r'$t$', fontsize=18)
ax2.set_ylabel(r'$\theta_2$', fontsize=18)
ax2.tick_params(labelsize=16)  # 刻度字体大小16
# 双摆问题的耦合
ax3.plot(odes_rk.ode_sol[:, 1], odes_rk.ode_sol[:, 3], 'c')
ax3.set_xlabel(r'$\theta_1$', fontsize=18)
ax3.set_ylabel(r'$\theta_2$', fontsize=18)
ax3.grid(ls=":")
ax3.tick_params(labelsize=16)  # 刻度字体大小16
plt.suptitle("双摆问题的$ODEs$数值解", fontsize=18)
plt.show()

# 角度变量与坐标的转换公式如下
theta1_np, theta2_np = odes_rk.ode_sol[:, 1], odes_rk.ode_sol[:, 3]
x1, y1 = 2 * np.sin(theta1_np), -2 * np.cos(theta1_np)  # 第1摆的长度为2
x2, y2 = x1 + np.sin(theta2_np), y1 - np.cos(theta2_np)

ax1, ax2, ax3 = ax_split()
ax1.plot(odes_rk.ode_sol[:, 0], x1, 'r', odes_rk.ode_sol[:, 0], y1, 'b')
ax1.set_ylabel(r'$x_1, y_1$', fontsize=18)
ax1.set_yticks([-3, 0, 3])
ax1.tick_params(labelsize=16)  # 刻度字体大小16
ax2.plot(odes_rk.ode_sol[:, 0], x2, 'r', odes_rk.ode_sol[:, 0], y2, 'b')
ax2.set_xlabel('$t$', fontsize=18)
ax2.set_ylabel(r'$x_2, y_2$', fontsize=18)
ax2.set_yticks([-3, 0, 3])
ax2.tick_params(labelsize=16)  # 刻度字体大小16
ax3.plot(x1, y1, 'r', lw=2.0, label="摆的轨迹1")
ax3.plot(x2, y2, 'b', lw=0.5, label="摆的轨迹2")
ax3.set_xlabel('$x$', fontsize=18)
ax3.set_ylabel('$y$', fontsize=18)
ax3.set_xticks([-3, 0, 3])
ax3.set_yticks([-3, 0, 3])
ax3.legend(frameon=False, fontsize=16)
plt.suptitle("双摆问题的动力学曲线", fontsize=18)
ax3.tick_params(labelsize=16)  # 刻度字体大小16
plt.show()
