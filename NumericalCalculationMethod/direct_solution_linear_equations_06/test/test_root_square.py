# -*- coding: UTF-8 -*-
"""
@file_name: test_root_square.py
@time: 2021-09-29
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from direct_solution_linear_equations_06.square_root_decomposition import SquareRootDecompositionAlgorithm

# A = np.array([[1, 2, 3, 4], [2, 5, 6, 7], [3, 6, 10, 9], [4, 7, 9, 27]])
# b = np.array([10, 17, 25, 60])
# A = np.array([[1, 2, 1, -3], [2, 5, 0, -5], [1, 0, 14, 1], [-3, -5, 1, 15]])
# b = np.array([2, 2, 15, -2])
A = np.array([[4, 1, -1, 0], [1, 3, -1, 0], [-1, -1, 5, 2], [0, 0, 2, 4]])
b = np.array([7, 8, -4, 6])

srda = SquareRootDecompositionAlgorithm(A, b, sol_method="cholesky")
srda.fit_solve()
print("下三角矩阵L为：\n", srda.L)
print("线性方程组的解x为：\n", srda.x)
print("解的精度验证：\n", srda.eps)
print("=" * 60)
srda = SquareRootDecompositionAlgorithm(A, b, sol_method="improved")
srda.fit_solve()
print("下三角矩阵L为：\n", srda.L)
print("对角矩阵D为：\n", srda.D)
print("线性方程组的解x为：\n", srda.x)
print("解的精度验证：\n", srda.eps)
