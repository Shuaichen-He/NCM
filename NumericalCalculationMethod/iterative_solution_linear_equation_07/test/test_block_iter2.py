# -*- coding: UTF-8 -*-
"""
@file_name: test.py
@time: 2022-12-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import pandas as pd
import numpy as np
from iterative_solution_linear_equation_07.block_iterative_method import BlockIterative
from iterative_solution_linear_equation_07.jacobi_gauss_seidel_iterative import JacobiGSlIterativeMethod
from util_font import *

Ab = pd.read_csv("../data/Ab.csv", header=None)
A = np.asarray(Ab.iloc[:, :-1])
b = np.asarray(Ab.iloc[:, -1])
x0 = np.ones(12)
eps = 1e-16
# 求解及可视化
plt.figure(figsize=(14, 5))
plt.subplot(121)
block_vector = np.ones(12)  # 快向量，既不使用块迭代法
# 2. 块Jacobi迭代法
b_jacobi = BlockIterative(A, b, x0, block_vector, eps=eps, is_out_info=True)
b_jacobi.fit_solve()
# 3. 块G-S迭代法
gs_block = BlockIterative(A, b, x0, block_vector, eps=eps, is_out_info=True, method="G-S")
gs_block.fit_solve()
# 4. 块SOR迭代法
sor_block = BlockIterative(A, b, x0, block_vector, eps=eps, is_out_info=True, omega=1.11, method="SOR")
sor_block.fit_solve()

iter_n_jacobi = b_jacobi.iterative_info["Iteration_number"]  # 获取雅可比迭代次数
iter_n_jacobi = np.linspace(1, iter_n_jacobi, iter_n_jacobi)
iter_n_gs = gs_block.iterative_info["Iteration_number"]  # 获取G-S迭代次数
iter_n_gs = np.linspace(1, iter_n_gs, iter_n_gs)
iter_n_sor = sor_block.iterative_info["Iteration_number"]  # 获取G-S迭代次数
iter_n_sor = np.linspace(1, iter_n_sor, iter_n_sor)
label_text = "$BSOR: \epsilon=%.2e, k=%d, \omega=1.11$" % (sor_block.precision[-1], iter_n_sor[-1])
plt.semilogy(iter_n_sor, sor_block.precision, "s-.", lw=1.5, label=label_text)
plt.semilogy(iter_n_jacobi, b_jacobi.precision, "o--", lw=1.5,
         label=r"$BJ: \epsilon=%.2e, k=%d$" % (b_jacobi.precision[-1], iter_n_jacobi[-1]))
plt.semilogy(iter_n_gs, gs_block.precision, "p-", lw=1.5,
         label="$BGS: \epsilon=%.2e, k=%d$" % (gs_block.precision[-1], iter_n_gs[-1])) # 绘制G-S

plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
plt.ylabel("$Precision(\epsilon=\Vert b - Ax^* \Vert _2)$", fontdict={"fontsize": 18})
plt.title("块迭代法的收敛曲线，块向量$[1,1,\cdots,1]$", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")

plt.subplot(122)
# block_vector = np.array([2, 2, 4, 2, 2])  # omega=1.15
block_vector = np.array([3, 2, 2, 2, 3])
# 2. 块Jacobi迭代法
b_jacobi = BlockIterative(A, b, x0, block_vector, eps=eps, is_out_info=True)
b_jacobi.fit_solve()
# 3. 块G-S迭代法
gs_block = BlockIterative(A, b, x0, block_vector, eps=eps, is_out_info=True, method="G-S")
gs_block.fit_solve()
# 4. 块SOR迭代法
sor_block = BlockIterative(A, b, x0, block_vector, eps=eps, is_out_info=True, omega=1.1, method="SOR")
sor_block.fit_solve()

iter_n_jacobi = b_jacobi.iterative_info["Iteration_number"]  # 获取雅可比迭代次数
iter_n_jacobi = np.linspace(1, iter_n_jacobi, iter_n_jacobi)
iter_n_gs = gs_block.iterative_info["Iteration_number"]  # 获取G-S迭代次数
iter_n_gs = np.linspace(1, iter_n_gs, iter_n_gs)
iter_n_sor = sor_block.iterative_info["Iteration_number"]  # 获取G-S迭代次数
iter_n_sor = np.linspace(1, iter_n_sor, iter_n_sor)
label_text = "$BSOR: \epsilon=%.2e, k=%d, \omega=1.10$" % (sor_block.precision[-1], iter_n_sor[-1])
plt.semilogy(iter_n_sor, sor_block.precision, "s-.", lw=1.5, label=label_text)
plt.semilogy(iter_n_jacobi, b_jacobi.precision, "o--", lw=1.5,
         label=r"$BJ: \epsilon=%.2e, k=%d$" % (b_jacobi.precision[-1], iter_n_jacobi[-1]))
plt.semilogy(iter_n_gs, gs_block.precision, "p-", lw=1.5,
         label="$BGS: \epsilon=%.2e, k=%d$" % (gs_block.precision[-1], iter_n_gs[-1])) # 绘制G-S

plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
plt.ylabel("$Precision(\epsilon=\Vert b - Ax^* \Vert _2)$", fontdict={"fontsize": 18})
plt.title("块迭代法的收敛曲线，块向量$[2,2,4,2,2]$", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.show()