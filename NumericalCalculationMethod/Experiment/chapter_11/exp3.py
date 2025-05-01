# -*- coding: UTF-8 -*-
"""
@file_name: exp3.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from ode_numerical_solution_11.linear_multi_step_method import LinearMultiStepMethod
from Experiment.util_font import *

ode_fun = lambda x, y: 1 / x ** 2 - y / x
analytic_ft = lambda x: (np.log(x) + 1) / x

plt.figure(figsize=(14, 5))
x0, y0, h_s = 1, 1, [0.01, 0.001]
for k, h in enumerate(h_s):
    xi = np.arange(1, 2 + h, h)
    sol_y = np.zeros((len(xi), 6))
    sol_y[:, 0] = analytic_ft(xi)  # 解析解
    precision = np.zeros((len(xi), 5))
    ode_methods = ["e_admas", "i_admas", "milne", "simpson", "hamming"]
    for idx, method in enumerate(ode_methods):
        ode_obj = LinearMultiStepMethod(ode_fun, x0, y0, x_final=2, h=h, ode_method=method)
        sol = ode_obj.fit_ode()
        sol_y[:, idx + 1] = sol[:, 1]
        precision[:, idx] = np.abs(sol_y[:, 0] - sol[:, 1])

    # 绘制与解析解对比的精度曲线
    plt.subplot(121 + k)
    line_style = ["-o", "-*", "-+", "-p", "-s"]
    labels = ["Adams E", "Adams I", "Milne E", "Simpson I", "Hamming I"]
    for i in range(5):
        error_norm = np.linalg.norm(precision[:, i])
        label_txt = labels[i] + ", \ \epsilon= %.2e" % error_norm
        if h == 0.05:
            plt.semilogy(xi, np.abs(precision[:, i]), line_style[i], lw=1.5, label="$%s$" % label_txt)
        else:
            plt.semilogy(xi, np.abs(precision[:, i]), lw=1.5, label="$%s$" % label_txt)
    plt.xlabel("$x$", fontdict={"fontsize": 18})
    plt.ylabel(r"$err = \vert y_k - \hat y_k \vert$", fontdict={"fontsize": 18})
    plt.title("线性多步法$ODE$数值解误差曲线 $h=%.1e$" % h, fontdict={"fontsize": 18})
    plt.legend(frameon=False, fontsize=16)
    plt.tick_params(labelsize=18)  # 刻度字体大小16
    plt.grid(ls=":")
plt.show()
