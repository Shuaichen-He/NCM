# -*- coding: UTF-8 -*-
"""
@file_name: exp4.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from Experiment.util_font import *
from ode_numerical_solution_11.first_order_ODEs_RK import FirstOrderODEsRK


def ode_funs(t, y):
    """
    微分方程组定义
    :param t: 时间变量
    :param y: 向量, y[0]表示x, y[1]表示y, y[2]表示z
    :return:
    """
    sigma, rou, beta = 16, 45, 4
    return np.array([sigma * (y[1] - y[0]),
                     y[0] * (rou - y[2]) - y[1],
                     y[0] * y[1] - beta * y[2]])


x0, y0, h = 0, np.array([12, 4, 1]), 0.001
fig = plt.figure(figsize=(14, 10))
x_finals = [20, 30, 40, 60]
for i, x_f in enumerate(x_finals):
    odes_rk = FirstOrderODEsRK(ode_funs, x0, y0, x_final=x_f, h=h)
    odes_rk.fit_odes()
    # 绘制洛伦兹方程曲线
    ax = fig.add_subplot(221 + i, projection='3d')
    ax.plot(odes_rk.ode_sol[:, 1], odes_rk.ode_sol[:, 2], odes_rk.ode_sol[:, 3])
    ax.set_xlabel("x", fontdict={"fontsize": 18})
    ax.set_ylabel("y", fontdict={"fontsize": 18})
    ax.set_zlabel("z", fontdict={"fontsize": 18})
    plt.title("$Lorenz \ Equation \ by \ Runge \ Kutta \ with \ [0, %d]$" % x_f, fontdict={"fontsize": 18})
    plt.tick_params(labelsize=18)  # 刻度字体大小18
plt.show()
