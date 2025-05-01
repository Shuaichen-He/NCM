# -*- coding: UTF-8 -*-
"""
@file_name: test_spring_dynamics_system.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from ode_numerical_solution_11.first_order_ODEs_RK import FirstOrderODEsRK


# 定义微分方程组
def ode_spring(t, y):
    m1, k1, g1 = 1.0, 10.0, 0.5  # 参数赋值
    m2, k2, g2 = 2.0, 40.0, 0.25  # 参数赋值
    dy = np.array([y[1], - k1 / m1 * y[0] + k2 / m1 * (y[2] - y[0]) - g1 / m1 * y[1],
                   y[3], - k2 / m2 * (y[2] - y[0]) - g2 / m2 * y[3]])
    return dy

x0, y0, h = 0, np.array([1.0, 0, 0.5, 0]), 0.001  # 初始值及求解微分步长
odes_rk = FirstOrderODEsRK(ode_spring, x0, y0, x_final=20, h=h)  # 实例化
odes_rk.fit_odes()  # 求解微分方程组

# 可视化，三个子图
fig = plt.figure(figsize=(12, 5))
ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=3)
ax2 = plt.subplot2grid((2, 5), (1, 0), colspan=3)
ax3 = plt.subplot2grid((2, 5), (0, 3), colspan=2, rowspan=2)
# 物体x1
ax1.plot(odes_rk.ode_sol[:, 0], odes_rk.ode_sol[:, 1], 'r')
ax1.set_ylabel('$x_1$', fontsize=18)
ax1.set_yticks([-1, -.5, 0, .5, 1])
ax1.tick_params(labelsize=14)  # 刻度字体大小16
# 物体x2
ax2.plot(odes_rk.ode_sol[:, 0], odes_rk.ode_sol[:, 3], 'g')
ax2.set_xlabel('$t$', fontsize=18)
ax2.set_ylabel('$x_2$', fontsize=18)
ax2.set_yticks([-1, -.5, 0, .5, 1])
ax2.tick_params(labelsize=14)  # 刻度字体大小16
# 物体x1和x2的关系
ax3.plot(odes_rk.ode_sol[:, 1], odes_rk.ode_sol[:, 3], 'c')
ax3.set_xlabel('$x_1$', fontsize=18)
ax3.set_ylabel('$x_2$', fontsize=18)
ax3.set_xticks([-1, -.5, 0, .5, 1])
ax3.set_yticks([-1, -.5, 0, .5, 1])
ax3.tick_params(labelsize=14)  # 刻度字体大小16
ax3.grid(ls=":")
plt.suptitle("龙格库塔法求解两个耦合的阻尼振荡器的$ODEs$解", fontsize=18)
plt.show()



