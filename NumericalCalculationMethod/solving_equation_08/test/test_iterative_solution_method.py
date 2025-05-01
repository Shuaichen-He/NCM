# -*- coding: UTF-8 -*-
"""
@file_name: test_iterative_solution_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from solving_equation_08.iterative_solution_method import IterativeSolutionMethod_Root
from util_font import *

fai_x1 = lambda x: np.sqrt(np.sqrt(x + 4) - 1)  # 迭代公式
fai_x2 = lambda x: (3 + x - 2 * x ** 2) ** (1 / 4)
fai_x3 = lambda x: -3 + 2 * x ** 2 + x ** 4
fai_x4 = lambda x: 2 * np.log(x) + np.log(3)
fai_x5 = lambda x: x ** 3 - 1
fai_x6 = lambda x: 1 / (2 - x) ** 2

method_eval = ["stable", "aitken", "steffensen"]
plt.figure(figsize=(14, 5))
plt.subplot(121)
line_stype = ["*-.", "o--", "s-"]
for method, style in zip(method_eval, line_stype):
    ism = IterativeSolutionMethod_Root(fai_x1, x0=1.0, eps=1e-16, max_iter=200, method=method)
    ism.fit_root()
    info = ism.root_precision_info
    plt.plot(info[:, 0], info[:, 2], style,
                 label="$%s: \ k=%d, \ \epsilon=%.2e$" % (method, len(info), info[-1, 2]))
plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
plt.title(r"各迭代法近似根$x^*$的$\epsilon=\vert x_{k+1} - x_k \vert$收敛曲线", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.subplot(122)
line_stype = ["*-.", "o--", "s-"]
for method, style in zip(method_eval, line_stype):
    ism = IterativeSolutionMethod_Root(fai_x1, x0=1.0, eps=1e-16, max_iter=200, method=method)
    ism.fit_root()
    info = ism.root_precision_info
    plt.semilogy(info[:, 0], info[:, 2], style,
                 label="$%s: \ k=%d, \ \epsilon=%.2e$" % (method, len(info), info[-1, 2]))
plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
plt.title(r"各迭代法近似根$x^*$的$\epsilon=\vert x_{k+1} - x_k \vert$收敛曲线", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.ylim([1e-17, 0.25])
plt.show()

plt.figure(figsize=(14, 5))
plt.subplot(121)
line_stype = ["*-.", "o--", "s-"]
for method, style in zip(method_eval, line_stype):
    ism = IterativeSolutionMethod_Root(fai_x2, x0=1.0, eps=1e-16, max_iter=200, method=method)
    ism.fit_root()
    info = ism.root_precision_info
    plt.semilogy(info[:40, 0], info[:40, 2], style,
                 label="$%s: \ k=%d, \ \epsilon=%.2e$" % (method, len(info), info[-1, 2]))
plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
plt.title(r"近似根$x^*$的$\epsilon=\vert x_{k+1} - x_k \vert$收敛曲线(前40次)", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.subplot(122)
ism = IterativeSolutionMethod_Root(fai_x3, x0=1, eps=1e-16, max_iter=200, method="steffensen")
ism.fit_root()
info = ism.root_precision_info
plt.semilogy(info[:, 0], info[:, 2], "s-", label="$steffensen: \ k=%d, \ \epsilon=%.2e$" % (len(info), info[-1, 2]))
plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
plt.title("$steffensen$求解迭代公式$(3)$的精度收敛曲线", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16, loc="lower left")
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.show()
