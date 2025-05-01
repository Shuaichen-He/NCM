# -*- coding: UTF-8 -*-
"""
@file_name: test_steepest_descent.py
@time: 2021-10-05
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from iterative_solution_linear_equation_07.steepest_descent_method import SteepestDescentMethod
from iterative_solution_linear_equation_07.jacobi_gauss_seidel_iterative import JacobiGSlIterativeMethod

# 求解例4方程组
# A = -1 * np.array([[-3, 1, 0, 0, 0, 0.5], [1, -3, 1, 0, 0, 0], [0, 1, -3, 1, 0, 0],
#                    [0, 0, 1, -3, 1, 0], [0, 0, 0, 1, -3, 1], [0.5, 0, 0, 0, 1, -3]])
# b = np.array([2.5, 1.5, 1, 1, 1.5, 2.5])
# x0 = np.zeros(6)

# 求解例5：构造三对角稀疏矩阵
n = 50  # 维度
A = np.zeros((n, n))
A = A + np.diag(4 * np.ones(n)) + np.diag( np.ones(n - 1), -1) + np.diag( np.ones(n - 1), 1)
b = 3 * np.ones(n)  # 问题(1)的右端向量
# x0 = 0.01 * np.ones(n)  # 初始解向量
x0 = np.zeros(n)  # 初始解向量
plt.figure(figsize=(14, 5))
sdm = SteepestDescentMethod(A, b, x0, eps=1e-15, is_out_info=True)
sdm.fit_solve()
plt.subplot(121)
sdm.plt_convergence_x(is_show=False, style="go-")

gs = JacobiGSlIterativeMethod(A, b, x0, eps=1e-15, is_out_info=True, method="G-S")
gs.fit_solve()
plt.subplot(122)
gs.plt_convergence(is_show=False)
plt.show()