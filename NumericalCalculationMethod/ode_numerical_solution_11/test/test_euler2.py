# -*- coding: UTF-8 -*-
"""
@file_name: test_euler.py
@time: 2021-11-09
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from ode_numerical_solution_11.ode_euler_method import ODEEulerMethod
from ode_numerical_solution_11.utils.ode_utils import ODESolUtils
from util_font import *


def ode_fun1(x, y):
    return -2 * x * y ** 2


ode_fun = lambda x, y: y - 2 * x / y
analytic_ft = lambda x: np.sqrt(1 + 2 * x)

ode_method = ["Explicit", "Implicit", "Trapezoid", "Middle", "PC"]
methods = ["Explicit", "Implicit", "Trapezoid", "Middle \ Point", "PC \ System"]

plt.figure(figsize=(14, 5))
plt.subplot(121)
h = 0.05
xi = np.arange(0, 1 + h, h)
print(len(xi))
yi = np.sqrt(1 + 2 * xi)  # 精确解
x0 = np.array([0.045, 0.4, 0.487, 0.685, 0.778, 0.923])  # 所求任意点的解
y0 = np.zeros((len(x0), 5))
line_style = [":", "--", "-.", "-", "-"]
precision = np.zeros((len(xi), 5))
for i, method in enumerate(ode_method):
    euler = ODEEulerMethod(ode_fun, x0=0, y0=1, x_final=1, h=h, ode_method=method)
    sol = euler.fit_ode()
    utils = ODESolUtils(euler, analytic_ft)
    y0[:, i] = utils.predict_x0(x0)  # 任意点的解
    precision[:, i] = yi - sol[:, 1]  # 精度误差
for i in range(5):
    error = np.linalg.norm(precision[:, i])
    label_txt = methods[i] + ", \ \epsilon= %.2e" % error
    plt.plot(xi[1:], precision[1:, i], line_style[i], lw=1.5, label="$%s$" % label_txt)
plt.xlabel("$x(h=0.05)$", fontdict={"fontsize": 18})
plt.ylabel(r"$err = y_k - \hat y_k$", fontdict={"fontsize": 18})
plt.title("$ODE$初值问题数值解的误差曲线 $\epsilon = \Vert y - \hat{y} \Vert _2$", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16, loc="upper left")
plt.grid(ls=":")
plt.tick_params(labelsize=18)  # 刻度字体大小16

# 输出区间任意点的数值解
y_true = np.sqrt(1 + 2 * x0)  # 精确解
for i in range(5):
    print(ode_method[i], end=": ")
    print(y0[:, i])
    print(y_true - y0[:, i])
    print("-" * 70)

plt.subplot(122)
ode_method = ["Trapezoid", "PC"]
methods = ["Trapezoid", "PC \ System"]
h = 0.0001
xi = np.arange(0, 1 + h, h)
print(len(xi))
yi = np.sqrt(1 + 2 * xi)  # 精确解
x0 = np.array([0.045, 0.4, 0.487, 0.685, 0.778, 0.923])  # 所求任意点的解
y0 = np.zeros((len(x0), 2))
line_style = ["--", "-"]
precision = np.zeros((len(xi), 2))
for i, method in enumerate(ode_method):
    euler = ODEEulerMethod(ode_fun, x0=0, y0=1, x_final=1, h=h, ode_method=method)
    sol = euler.fit_ode()
    utils = ODESolUtils(euler, analytic_ft)
    y0[:, i] = utils.predict_x0(x0)  # 任意点的解
    precision[:, i] = yi - sol[:, 1]  # 精度误差
# 绘制与解析解对比的精度曲线
for i in range(2):
    error = np.linalg.norm(precision[:, i])
    label_txt = methods[i] + ", \ \epsilon= %.2e" % error
    plt.semilogy(xi[1:], np.abs(precision[1:, i]), line_style[i], lw=1.5, label="$%s$" % label_txt)
plt.xlabel("$x(h=0.0001)$", fontdict={"fontsize": 18})
plt.ylabel(r"$err = \vert y_k - \hat y_k \vert$", fontdict={"fontsize": 18})
plt.title("$ODE$初值问题数值解的误差曲线 $\epsilon = \Vert y - \hat{y} \Vert _2$", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.grid(ls=":")
plt.tick_params(labelsize=18)  # 刻度字体大小16
plt.show()
