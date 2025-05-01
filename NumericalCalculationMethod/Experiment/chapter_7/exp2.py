# -*- coding: UTF-8 -*-
"""
@file_name: exp2.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from iterative_solution_linear_equation_07.block_iterative_method import BlockIterative
from Experiment.util_font import *  # 导入字体文件

A = np.array([[2.6934, 0.6901, 0.3997, 0.6010, 0.4390], [0.6901, 2.8784, 0.8799, 0.5978, 0.4514],
              [0.3997, 0.8799, 3.3216, 0.4673, 0.8282], [0.6010, 0.5978, 0.4673, 2.8412, 0.5511],
              [0.4390, 0.4514, 0.8282, 0.5511, 2.8704]])  # 系数矩阵
b, x0 = np.array([1, 1, 1, 1, 1]), np.array([0, 0, 0, 0, 0])  # 右端向量，初始解向量

block_vectors = [np.array([4, 1]), np.array([2, 2, 1])]  # 块迭代向量
plt.figure(figsize=(14, 5))
for i, block_vector in enumerate(block_vectors):
    # 块Jacobi迭代法
    b_jacobi = BlockIterative(A, b, x0, block_vector, eps=1e-15, is_out_info=True)
    b_jacobi.fit_solve()
    # 块G-S迭代法
    gs_block = BlockIterative(A, b, x0, block_vector, eps=1e-15, is_out_info=True, method="G-S")
    gs_block.fit_solve()
    # 块SOR迭代法
    sor_block = BlockIterative(A, b, x0, block_vector, eps=1e-15, is_out_info=True, omega=1.03, method="SOR")
    sor_block.fit_solve()
    print("=" * 80)
    # 可视化
    plt.subplot(121 + i)
    iter_n_jacobi = b_jacobi.iterative_info["Iteration_number"]  # 获取雅可比迭代次数
    iter_n_jacobi = np.linspace(1, iter_n_jacobi, iter_n_jacobi)
    iter_n_gs = gs_block.iterative_info["Iteration_number"]  # 获取G-S迭代次数
    iter_n_gs = np.linspace(1, iter_n_gs, iter_n_gs)
    iter_n_sor = sor_block.iterative_info["Iteration_number"]  # 获取G-S迭代次数
    iter_n_sor = np.linspace(1, iter_n_sor, iter_n_sor)
    label_text = "$BSOR: \epsilon=%.2e, k=%d, \omega=1.03$" % (sor_block.precision[-1], iter_n_sor[-1])
    plt.semilogy(iter_n_sor, sor_block.precision, "s-.", lw=1.5, label=label_text)
    plt.semilogy(iter_n_jacobi, b_jacobi.precision, "o--", lw=1.5,
                 label=r"$BJ: \epsilon=%.2e, k=%d$" % (b_jacobi.precision[-1], iter_n_jacobi[-1]))
    plt.semilogy(iter_n_gs, gs_block.precision, "p-", lw=1.5,
                 label="$BGS: \epsilon=%.2e, k=%d$" % (gs_block.precision[-1], iter_n_gs[-1]))  # 绘制G-S

    plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
    plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
    if i == 0:
        plt.title("块迭代法的$\epsilon=\Vert b - Ax^* \Vert _2$收敛曲线，块向量$[4,1]$",
                  fontdict={"fontsize": 18})
    else:
        plt.title("块迭代法的$\epsilon=\Vert b - Ax^* \Vert _2$收敛曲线，块向量$[2,2,1]$",
                  fontdict={"fontsize": 18})
    plt.legend(frameon=False, fontsize=16)
    plt.tick_params(labelsize=16)  # 刻度字体大小16
    plt.grid(ls=":")
plt.show()
