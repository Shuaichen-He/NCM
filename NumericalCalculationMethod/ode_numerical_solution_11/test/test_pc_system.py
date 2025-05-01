# -*- coding: UTF-8 -*-
"""
@file_name: test_pc_system.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from ode_numerical_solution_11.predictive_correction_system import PredictiveCorrectionSystem
from util_font import *

ode_fun = lambda x, y: 2 * y / x + x ** 2 * np.exp(x)  # ODE方程
plt.figure(figsize=(14, 5))
x0, y0, h_s = 1, 0, [0.05, 0.001]
for k, h in enumerate(h_s):
    xi = np.arange(1, 2 + h, h)
    sol_y = np.zeros((len(xi), 4))
    sol_y[:, 0] = xi ** 2 * (np.exp(xi) - np.exp(1))  # 解析解
    precision = np.zeros((len(xi), 3))
    pc_forms, label_method = ["PECE", "PMECME_A", "PMECME_MH"], ["PECE", "PMECME-A", "PMECME-MH"]
    for idx, form in enumerate(pc_forms):
        pcs = PredictiveCorrectionSystem(ode_fun, x0, y0, x_final=2, h=h, pc_form=form)
        sol = pcs.fit_ode()
        sol_y[:, idx + 1] = sol[:, 1]
        precision[:, idx] = np.abs(sol_y[:, 0] - sol[:, 1])
        print(precision[:, idx])

    # 绘制与解析解对比的精度曲线
    line_style = ["--o", "-*", "-.+"]
    plt.subplot(121 + k)
    for i in range(3):
        error_norm = np.linalg.norm(precision[:, i])
        label_txt = label_method[i] + ", \ \epsilon= %.5e" % error_norm
        if h == 0.05:
            plt.semilogy(xi[1:], np.abs(precision[1:, i]), line_style[i], lw=1.5, label="$%s$" % label_txt)
        else:
            plt.semilogy(xi[1:], np.abs(precision[1:, i]), lw=1.5, label="$%s$" % label_txt)
    plt.xlabel("$x$", fontdict={"fontsize": 18})
    plt.ylabel(r"$err = \vert y_k - \hat y_k \vert$", fontdict={"fontsize": 18})
    plt.title("预测校正系统$ODE$数值解误差曲线 $h=%.3f$" % h, fontdict={"fontsize": 18})
    plt.legend(frameon=False, fontsize=16)
    plt.tick_params(labelsize=18)  # 刻度字体大小16
    plt.grid(ls=":")
plt.show()
