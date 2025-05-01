# -*- coding: UTF-8 -*-
"""
@file_name: test_odes_rk.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from ode_numerical_solution_11.first_order_ODEs_RK import FirstOrderODEsRK


def ode_funs1(x, y):  # 定义微分方程组
    return np.array([y[1],  # 表示方程1: y'_1(x)=y_2(x)
                     5 * np.exp(2 * x) * np.sin(x) - 2 * y[0] + 2 * y[1]])


def ode_funs2(x, y):
    return np.array([y[1], 5 * (1 - y[0] ** 2) * y[1] - y[0]])


def ode_funs3(x, y):
    dy = np.array([-0.013 * y[0] - 1000 * y[0] * y[1], -2500 * y[1] * y[2],
                   -0.013 * y[0] - 1000 * y[0] * y[1] - 2500 * y[1] * y[2]])
    return dy


def ode_funs4(x, y):
    k1, k2 = 1, 10
    return np.array([-k1 * y[0], k1 * y[0] - k2 * y[1], -k2 * y[1]])


x0, y0, h = 0, np.array([-2, -3]), 0.001  # 方程1
# x0, y0, h = 0, np.array([1, 2]), 0.001  # 方程2
# x0, y0, h = 0, np.array([1, 1, 0]), 0.0001  # 方程3
# x0, y0, h = 0, np.array([1, 1, 1]), 0.0001  # 方程4

xi = np.arange(0, 1 + h, h)
yi = np.exp(2 * xi) * (np.sin(xi) - 2 * np.cos(xi))
dyi = np.exp(2 * xi) * (4 * np.sin(xi) - 3 * np.cos(xi))
print("解析解：", yi)
print("一阶导数解析解：", dyi)
odes_rk = FirstOrderODEsRK(ode_funs1, x0, y0, x_final=1, h=h)
odes_rk.fit_odes()
print("数值解：", odes_rk.ode_sol[:, 1:])
# odes_rk.plt_odes_rk()  # 不存在解析解，则直接绘制数值解即可，对应微分方程组（2）

precision = np.abs(yi - odes_rk.ode_sol[:, 1])
d_precision = np.abs(dyi - odes_rk.ode_sol[:, 2])
print("误差精度：", precision)
print("一阶导数误差精度：", d_precision)
plt.figure(figsize=(14, 5))
plt.subplot(121)
odes_rk.plt_odes_rk(is_show=False)
plt.subplot(122)
line_style = ["-", "--", "-.", ":"]
plt.plot(odes_rk.ode_sol[:, 0], precision, "-", lw=1.5,
         label="$\hat y_1(x), \ \epsilon=%.7e$" % np.linalg.norm(precision))
plt.plot(odes_rk.ode_sol[:, 0], d_precision, "--", lw=1.5,
         label="$\hat y_2(x), \ \epsilon=%.7e$" % np.linalg.norm(d_precision))
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$err = \vert y_k - \hat y_k \vert$", fontdict={"fontsize": 18})
plt.title("龙格库塔法$ODEs$数值解误差曲线 $\epsilon = \Vert y - \hat{y} \Vert_2$", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=18)
plt.grid(ls=":")
plt.tick_params(labelsize=18)  # 刻度字体大小16
plt.show()
