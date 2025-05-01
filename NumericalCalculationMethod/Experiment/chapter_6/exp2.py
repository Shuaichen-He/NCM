# -*- coding: UTF-8 -*-
"""
@file_name: exp2.py
@time: 2023-02-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from direct_solution_linear_equations_06.square_root_decomposition import SquareRootDecompositionAlgorithm
from direct_solution_linear_equations_06.doolittle_decomposition_lu import DoolittleTriangularDecompositionLU

A = np.array([[4, 1, -1, 0], [1, 3, -1, 0], [-1, -1, 5, 2], [0, 0, 2, 4]])
b = np.array([7, 8, -4, 6])

# 矩阵三角分解法
dtd_lu = DoolittleTriangularDecompositionLU(A, b, sol_method="pivot")
dtd_lu.fit_solve()
print("下三角矩阵L：\n", dtd_lu.L)
print("上三角矩阵U：\n", dtd_lu.U)
print("置换矩阵P：\n", dtd_lu.P)
print("系数矩阵的逆矩阵：\n", dtd_lu.inverse_matrix)
print("逆矩阵验证：", np.dot(A, dtd_lu.inverse_matrix))
print("线性方程组的解：", dtd_lu.x)
print("解的精度验证：", dtd_lu.eps)
print("=" * 80)

# 平方根法
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