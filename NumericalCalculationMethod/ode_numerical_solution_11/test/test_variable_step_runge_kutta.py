# -*- coding: UTF-8 -*-
"""
@file_name: test_runge_kuta.py
@time: 2021-11-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from ode_numerical_solution_11.variable_step_runge_kutta import VariableStepRungeKutta
from ode_numerical_solution_11.utils.ode_utils import ODESolUtils
from util_font import *

ode_fun = lambda x, y: -50 * y + 50 * x ** 2 + 2 * x
analytic_ft = lambda x: 1 / 3 * np.exp(-50 * x) + x ** 2
# x0 = np.array([0.345, 0.8459, 1.4, 1.787, 2.285, 2.678, 2.923, 3.0785])  # 所求任意点的解
x0 = np.array([0.045, 0.2459, 0.4, 0.487, 0.685, 0.778, 0.923, 0.9785])  # 所求任意点的解
y0 = np.zeros((len(x0), 2))
rk = VariableStepRungeKutta(ode_fun, x0=0, y0=1/3, x_final=1, h=0.001, eps=1e-12)
sol = rk.fit_ode()
utils = ODESolUtils(rk, analytic_ft)
y_pred = utils.predict_x0(x0)
y_true = analytic_ft(x0)  # 精确解
print("预测值：\n", y_pred)
print("误差：\n", y_true - y_pred)
plt.figure(figsize=(14, 5))
plt.subplot(121)
rk.plt_histogram_dist(is_show=False)

yi = analytic_ft(sol[:, 0])  # 精确解
# 绘制与解析解对比的精度曲线
plt.subplot(122)
precision = yi - sol[:, 1]
error = np.linalg.norm(precision)
plt.semilogy(sol[:, 0], np.abs(precision), lw=1.5, label="$\epsilon = %.7e$" % error)
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$err = \vert y_k - \hat y_k \vert$", fontdict={"fontsize": 18})
plt.title("求解$ODE$数值解的误差曲线 $\epsilon = \Vert y - \hat{y} \Vert_2$", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.grid(ls=":")
plt.tick_params(labelsize=18)  # 刻度字体大小16
# plt.ylim([1e-14, 1e-12])
plt.show()
