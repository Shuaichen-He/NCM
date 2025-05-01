# -*- coding: UTF-8 -*-
"""
@file_name: test_jacobi_gs.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from iterative_solution_linear_equation_07.jacobi_gauss_seidel_iterative import JacobiGSlIterativeMethod
from util_font import *

# 构造带状稀疏矩阵，例1示例
n = 50  # 维度
A = np.zeros((n, n))
A = A + np.diag(12 * np.ones(n))  # 主对角线
A = A + np.diag(-2 * np.ones(n - 1), -1)  # 下次对角线
A = A + np.diag(-2 * np.ones(n - 1), 1)  # 上次对角线
A = A + np.diag(np.ones(n - 2), -2)  # 下次对角线
A = A + np.diag(np.ones(n - 2), 2)  # 上次对角线
b = 5 * np.ones(n)  # 右端向量
x0 = 0.01 * np.ones(n)  # 初始解向量

jacobi = JacobiGSlIterativeMethod(A, b, x0, eps=1e-14, method="Jacobi", is_out_info=True)
jacobi.fit_solve()  # 雅可比迭代求解
print("=" * 50)
guass_seidel = JacobiGSlIterativeMethod(A, b, x0, eps=1e-14, method="G-S", is_out_info=True)
guass_seidel.fit_solve()  # G-S迭代求解

plt.figure(figsize=(14, 5))
iter_num_jacobi = jacobi.iterative_info["Iteration_number"]  # 获取雅可比迭代次数
iter_num_jacobi = np.linspace(1, iter_num_jacobi, iter_num_jacobi)
iter_num_gs = guass_seidel.iterative_info["Iteration_number"]  # 获取G-S迭代次数
iter_num_gs = np.linspace(1, iter_num_gs, iter_num_gs)
plt.subplot(121)
plt.semilogy(iter_num_jacobi, jacobi.precision, "o--", lw=1.5,
             label=r"$Jacobi: \ \epsilon=%.3e,\  k=%d$" % (jacobi.precision[-1], iter_num_jacobi[-1]))
plt.semilogy(iter_num_gs, guass_seidel.precision, "*-", lw=1.5,
             label="$G-S: \ \epsilon=%.3e, \ k=%d$" % (guass_seidel.precision[-1], iter_num_gs[-1]))  # 绘制G-S
plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
plt.title("$Jacobi$和$Gauss-Seidel$的$\epsilon=\Vert b - Ax^* \Vert _2$收敛曲线", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.subplot(122)
plt.plot(iter_num_jacobi, jacobi.precision, "o--", lw=1.5,
         label=r"$Jacobi: \ \epsilon=%.3e,\  k=%d$" % (jacobi.precision[-1], iter_num_jacobi[-1]))
plt.plot(iter_num_gs, guass_seidel.precision, "*-", lw=1.5,
         label="$G-S: \ \epsilon=%.3e, \ k=%d$" % (guass_seidel.precision[-1], iter_num_gs[-1]))  # 绘制G-S
plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
plt.title("$Jacobi$和$Gauss-Seidel$的$\epsilon=\Vert b - Ax^* \Vert _2$收敛曲线", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.show()
