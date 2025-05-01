# -*- coding: UTF-8 -*-
"""
@file_name: test_runge_kuta.py
@time: 2021-11-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from ode_numerical_solution_11.ode_runge_kutta_method import ODERungeKuttaMethod
from ode_numerical_solution_11.utils.ode_utils import ODESolUtils
from util_font import *


def ode_fun(x, y):
    return -50 * y + 50 * x ** 2 + 2 * x

def ode_fun2(x, y):
    return np.cos(2 * x) + np.sin(3 * x)

def ode_fun3(x, y):
    return x * np.exp(3 * x) - 2 * y

analytic_ft = lambda x: 1 / 3 * np.exp(-50 * x) + x ** 2
# analytic_ft2 = lambda x: 1 / 2 * np.sin(2 * x) - 1 / 3 * np.cos(3 * x) + 4 / 3
# analytic_ft3 = lambda x: 0.2 * x * np.exp(3 * x) - 1 / 25 * np.exp(3 * x) + 1 / 25 * np.exp(-2 * x)


ode_method = ["RK", "RKF"]
plt.figure(figsize=(14, 5))
plt.subplot(121)
h = 0.001
xi = np.arange(0, 1 + h, h)
# yi = analytic_ft(xi)  # 精确解
yi = analytic_ft(xi)
x0 = np.array([0.045, 0.2459, 0.4, 0.487, 0.685, 0.778, 0.923, 0.9785])  # 所求任意点的解
y0 = np.zeros((len(x0), 2))
line_style, markers = ["-", "--"], ["o", "s"]
precision = np.zeros((len(xi), 2))
plt.plot(xi, yi, "k-", lw=1.5, label="$Exact \ sol \ y(x)$")
for i, method in enumerate(ode_method):
    rk = ODERungeKuttaMethod(ode_fun, x0=0, y0=1/3, x_final=1, h=h, rk_type=method)
    sol = rk.fit_ode()
    utils = ODESolUtils(rk, analytic_ft)
    y0[:, i] = utils.predict_x0(x0)  # 任意点的解
    precision[:, i] = yi - sol[:, 1]  # 精度误差
    plt.plot(sol[:, 0], sol[:, 1], line_style[i], lw=2, label="$%s \ \hat y(x)$" % method)
plt.plot(x0, y0[:, 0], "o", label="$RK \ (x_0, \hat y(x_0))$")
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$y(x) \ / \ \hat y(x)$", fontdict={"fontsize": 18})
plt.title("龙格库塔法求解$ODE$初值问题数值解曲线", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=18)
plt.grid(ls=":")
plt.tick_params(labelsize=18)  # 刻度字体大小16

# 绘制与解析解对比的精度曲线
plt.subplot(122)
for i in range(2):
    error = np.linalg.norm(precision[:, i])
    label_txt = ode_method[i] + ", \ \epsilon= %.5e" % error
    plt.semilogy(xi[1:], np.abs(precision[1:, i]), line_style[i], lw=2, label="$%s$" % label_txt)
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$err = \vert y_k - \hat y_k \vert$", fontdict={"fontsize": 18})
plt.title("$ODE$初值问题数值解的误差曲线 $\epsilon = \Vert y - \hat{y} \Vert _2$", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=18)
plt.grid(ls=":")
plt.tick_params(labelsize=18)  # 刻度字体大小16
# plt.ylim([2e-11, 1e-8])
plt.show()
# 输出区间任意点的数值解
y_true = analytic_ft(x0)  # 精确解
for i in range(2):
    print(ode_method[i], end=": ")
    print(y0[:, i])
    print(y_true - y0[:, i])
    print("-" * 70)
