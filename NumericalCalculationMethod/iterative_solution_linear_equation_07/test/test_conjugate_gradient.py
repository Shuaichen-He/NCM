# -*- coding: UTF-8 -*-
"""
@file_name: test_conjugate_gradient.py
@time: 2021-10-05
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from iterative_solution_linear_equation_07.conjugate_gradient_method import ConjugateGradientMethod
from iterative_solution_linear_equation_07.steepest_descent_method import SteepestDescentMethod

# A = np.array([[1, 1, 1, 1, 1], [1, 2, 3, 4, 5], [1, 3, 6, 10, 15],
#               [1, 4, 10, 20, 35], [1, 5, 15, 35, 70]])  # 系数矩阵
# b = np.array([5, 15, 35, 70, 126])  # 右端向量
# x0 = np.array([0, 0, 0, 0, 0])  # 初始解向量
# A = np.array([[-4, 1, 1, 1], [1, -4, 1, 1], [1, 1, -4, 1], [1, 1, 1, -4]])
# b = np.array([1, 1, 1, 1])
# x0 = np.array([0, 0, 0, 0])  # 初始解向量
# A = np.array([[9, 0, 1.2], [0.36, 10, -1.5], [-2.2, 0.72, 8]])
# b = np.array([1, 1, 1])
# x0 = np.array([0, 0, 0])
# A = np.array([[3, 1], [1, 2]])
# b = np.array([5, 5])
# x0 = np.array([0, 0])
# A = np.array([[3, 2, 0], [2, 4, -2], [0, -2, 5]])
# b = np.array([1, 1, 1])
# x0 = np.array([0, 0, 0])
# A = -1 * np.array([[-3, 1, 0, 0, 0, 0.5], [1, -3, 1, 0, 0, 0], [0, 1, -3, 1, 0, 0],
#               [0, 0, 1, -3, 1, 0], [0, 0, 0, 1, -3, 1], [0.5, 0, 0, 0, 1, -3]])
# b = np.array([2.5, 1.5, 1, 1, 1.5, 2.5])
# x0 = np.zeros(6)

#例5
# A = np.array([[4, 3, 0], [3, 4, -1], [0, -1, 4]])
# b = np.array([3, 5, -5])
# x0 = np.array([0, 0, 0])

n = 50  # 维度
A = np.zeros((n, n))
A = A + np.diag(4 * np.ones(n)) + np.diag(np.ones(n - 1), -1) + np.diag(np.ones(n - 1), 1)
b = 3 * np.ones(n)  # 右端向量
# idx = np.linspace(0, 49, 50, dtype=np.int)
# b[np.mod(idx, 2) == 0] = 1  # 偶数索引为1，奇数为2
x0 = 0.01 * np.ones(n)  # 初始解向量

cgm = ConjugateGradientMethod(A, b, x0, eps=1e-15, is_out_info=True)
cgm.fit_solve()
sdm = SteepestDescentMethod(A, b, x0, eps=1e-15, is_out_info=True)
sdm.fit_solve()
plt.figure(figsize=(14, 5))
plt.subplot(121)
sdm.plt_convergence_x(is_show=False, style="go-")
plt.subplot(122)
cgm.plt_convergence_x(is_show=False)
plt.show()
