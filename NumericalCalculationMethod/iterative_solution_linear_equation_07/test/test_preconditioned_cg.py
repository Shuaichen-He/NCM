# -*- coding: UTF-8 -*-
"""
@file_name: test_preconditioned_cg.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from iterative_solution_linear_equation_07.pre_conjugate_gradient import PreConjugateGradient
from iterative_solution_linear_equation_07.conjugate_gradient_method import ConjugateGradientMethod
from iterative_solution_linear_equation_07.steepest_descent_method import SteepestDescentMethod
from util_font import *

# A = np.array([[4, 3, 0], [3, 4, -1], [0, -1, 4]])
# b = np.array([3, 5, -5])
# x0 = np.array([0, 0, 0])

A = np.diag(np.array([1, 10, 100, 1000, 10000, 100000]))
b = np.array([1, 2, 3, 4, 5, 6])
x0 = np.array([0, 0, 0, 0, 0, 0])
pcg = PreConjugateGradient(A, b, x0, eps=1e-16, is_out_info=True)
pcg.fit_solve()
pcg_iter = pcg.iterative_info["Iteration_number"]
# pcg.plt_convergence_x()
print("-" * 50)
cgm = ConjugateGradientMethod(A, b, x0, eps=1e-16, is_out_info=True)
cgm.fit_solve()
cg_iter = cgm.iterative_info["Iteration_number"]

plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.semilogy(range(1, pcg_iter + 1), pcg.precision, "o-", lw=2,
             label="$PCG: \epsilon=%.2e, \ k=%d$" % (pcg.precision[-1], pcg_iter))
plt.semilogy(range(1, cg_iter + 1), cgm.precision, "*--", lw=2,
             label="$CG: \epsilon=%.2e, \ k=%d$" % (cgm.precision[-1], cg_iter))
plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
plt.title("$PCG$与$CG$的$\epsilon=\Vert b - Ax^* \Vert _2$收敛曲线", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=18)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")

plt.subplot(122)
A = np.array([[1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6], [1, 3, 6, 10, 15, 21],
              [1, 4, 10, 20, 35, 56], [1, 5, 15, 35, 70, 126], [1, 6, 21, 56, 126, 252]])  # 系数矩阵
b = np.array([6, 21/ 2, 14, 63 / 4, 63 / 4, 113 / 8])  # 右端向量
x0 = np.array([0, 0, 0, 0, 0, 0])  # 初始解向量
# n = 5
# A = 1. / (np.arange(1, n + 1) + np.arange(0, n)[:, np.newaxis])
# b = 0.01 * np.ones(n)  # 右端向量
# x0 = np.zeros(n)  # 初始解向量

pcg = PreConjugateGradient(A, b, x0, eps=1e-16, max_iter=10000, is_out_info=True)
pcg.fit_solve()
pcg_iter = pcg.iterative_info["Iteration_number"]
# pcg.plt_convergence_x()
print("-" * 50)
cgm = ConjugateGradientMethod(A, b, x0, eps=1e-16, max_iter=10000, is_out_info=True)
cgm.fit_solve()
cg_iter = cgm.iterative_info["Iteration_number"]
plt.semilogy(range(1, pcg_iter + 1), pcg.precision, "o-", lw=2,
             label="$PCG: \epsilon=%.2e, k=%d$" % (pcg.precision[-1], pcg_iter))
plt.semilogy(range(1, cg_iter + 1), cgm.precision, "*--", lw=2,
             label="$CG: \epsilon=%.2e, k=%d$" % (cgm.precision[-1], cg_iter))
plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
plt.title("$PCG$与$CG$的$\epsilon=\Vert b - Ax^* \Vert _2$收敛曲线", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=18)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.show()
