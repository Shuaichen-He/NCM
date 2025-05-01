# -*- coding: UTF-8 -*-
"""
@file_name: test_doolittle_lu.py
@time: 2021-09-28
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from direct_solution_linear_equations_06.doolittle_decomposition_lu import DoolittleTriangularDecompositionLU

A = np.array([[2, 5, 4, 1], [1, 3, 2, 1], [2, 10, 9, 7], [3, 8, 9, 2]])
b = np.array([20, 11, 40, 37])
# A = np.array([[3, -1, -2, -1], [-1, 1, 2, -1], [2, 2, -1, 3], [1, 1, 3, -2]])
# b = np.array([2, 4, -8, 10])
dtd_lu = DoolittleTriangularDecompositionLU(A, b)
dtd_lu.fit_solve()
print("下三角矩阵L：\n", dtd_lu.L)
print("上三角矩阵U：\n", dtd_lu.U)
# print(dtd_lu.P)
print("线性方程组的解：", dtd_lu.x)
print("解的精度验证：", dtd_lu.eps)
print("=" * 60)

# A = np.array([[1.0303, 0.99030], [0.99030, 0.95285]])
# b = np.array([2.4944, 2.3988])

# A = np.array([[10, 7, 8, 7], [7, 5, 6, 5], [8, 6, 10, 9], [7, 5, 9, 10]])
# A = np.array([[10, 7, 8.1, 7.2], [7.08, 5.04, 6, 5], [8, 5.98, 9.89, 9], [6.99, 5, 9, 9.98]])
# b = np.array([32, 23, 33, 31])
# A = np.array([[4, 1, -1, 0], [1, 3, -1, 0], [-1, -1, 5, 2], [0, 0, 2, 4]])
# b = np.array([7, 8, -4, 6])
A = np.array([[2, 5, 4, 1], [1, 3, 2, 1], [2, 10, 9, 7], [3, 8, 9, 2]])
b = np.array([20, 11, 40, 37])
dtd_lu = DoolittleTriangularDecompositionLU(A, b, sol_method="pivot")
dtd_lu.fit_solve()
print("下三角矩阵L：\n", dtd_lu.L)
print("上三角矩阵U：\n", dtd_lu.U)
print("置换矩阵P：\n", dtd_lu.P)
print("系数矩阵的逆矩阵：\n", dtd_lu.inverse_matrix)
print("逆矩阵验证：", np.dot(A, dtd_lu.inverse_matrix))
print("线性方程组的解：", dtd_lu.x)
print("解的精度验证：", dtd_lu.eps)
