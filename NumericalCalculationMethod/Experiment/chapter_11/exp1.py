# -*- coding: UTF-8 -*-
"""
@file_name: exp1.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from ode_numerical_solution_11.ode_euler_method import ODEEulerMethod
from ode_numerical_solution_11.utils.ode_utils import ODESolUtils
from Experiment.util_font import *


def ode_fun(x, y):
    """
    微分方程定义
    """
    return -y + x + 1


analytic_ft = lambda x: np.exp(-x) + x  # 解析解

ode_method = ["Trapezoid", "PC"]
methods = ["Trapezoid", "PC \ System"]
plt.figure(figsize=(14, 5))
plt.subplot(121)
h = 0.0001
xi = np.arange(0, 1 + h, h)
yi = analytic_ft(xi)  # 精确解
x0 = np.array([0.045, 0.4, 0.487, 0.685, 0.778, 0.923])  # 所求任意点的解
y0 = np.zeros((len(x0), 2))
line_style = ["--", "-."]
precision = np.zeros((len(xi), 2))
for i, method in enumerate(ode_method):
    euler = ODEEulerMethod(ode_fun, x0=0, y0=1, x_final=1, h=h, ode_method=method)
    sol = euler.fit_ode()
    utils = ODESolUtils(euler, analytic_ft)
    y0[:, i] = utils.predict_x0(x0)  # 任意点的解
    precision[:, i] = yi - sol[:, 1]  # 精度误差
    plt.plot(sol[:, 0], sol[:, 1], line_style[i], lw=1.5, label="$%s \ \hat y(x)$" % methods[i])
plt.plot(xi, yi, "k-", lw=1.5, label="$Exact \ sol \ y(x)$")
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$y(x) \ / \ \hat y(x)$", fontdict={"fontsize": 18})
plt.title("欧拉法求解$ODE$初值问题数值解曲线", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16, loc="best")
plt.grid(ls=":")
plt.tick_params(labelsize=18)  # 刻度字体大小16

# 绘制与解析解对比的精度曲线
plt.subplot(122)
for i in range(2):
    error = np.linalg.norm(precision[:, i])
    label_txt = methods[i] + ", \ \epsilon= %.2e" % error
    plt.plot(xi, precision[:, i], line_style[i], lw=1.5, label="$%s$" % label_txt)
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$err = y_k - \hat y_k$", fontdict={"fontsize": 18})
plt.title("$ODE$初值问题数值解的误差曲线 $\epsilon = \Vert y - \hat{y} \Vert$", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16, loc="best")
plt.grid(ls=":")
plt.tick_params(labelsize=18)  # 刻度字体大小16
plt.show()

# 输出区间任意点的数值解
y_true = analytic_ft(x0)  # 精确解
print("真值：", y_true)
for i in range(2):
    print(ode_method[i], end=": ")
    print(y0[:, i])
    print(y_true - y0[:, i])
    print("-" * 70)
