# -*- coding: UTF-8 -*-
"""
@file_name: test_qr_orthogonal.py
@time: 2021-09-30
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from direct_solution_linear_equations_06.qr_orthogonal_decomposition import QROrthogonalDecomposition


# A = np.array([[8, 3, 2], [4, 9, 1], [2, 6, 10]])
# b = np.array([1, 1, 1])
# A = np.array([[2, 5, 4, 1], [1, 3, 2, 1], [2, 10, 9, 7], [3, 8, 9, 2]])
# b = np.array([20, 11, 40, 37])
# A = np.array([[1, 2, 2], [1, 0, 2], [0, 1, 1]])
# b = np.array([1, 1, 1])
# A = np.array([[0, 3, 1], [0, 4, -2], [2, 1, 1]])
# b = np.array([1, 1, 1])
# A = np.array([[1, 1, 1], [2, 3, 1], [2, 1, -2]])
# b = np.array([1, 1, 1])
# qr = QROrthogonalDecomposition(A, b)
# qr.fit_solve()
# print("正交化矩阵Q：\n", qr.Q)
# print("上三角矩阵R：\n", qr.R)
# print("线性方程组的解：\n", qr.x)
# print("验证解的精度：\n", qr.eps)
# print(np.linalg.qr(A))

# A = np.array([[1, 1, 1], [2, 3, 1], [2, 1, -5]])
# b = np.array([1, 1, 1])

# A = np.array([[1, 2, 1, -3], [2, 5, 0, -5], [1, 0, 14, 1], [-3, -5, 1, 15]])
# b = np.array([2, 2, 15, -2])
A = np.array([[1, 3, 5, -4, 2], [1, 3, 2, -2, 1], [1, -2, 1, -1, -1], [1, -4, 1, 1, -1], [1, 2, 1, -1, 1]])
print(np.linalg.matrix_rank(A))
b = np.array([3, -1, 3, 3, -1])

sol_method = ["schmidt", "householder", "givens"]
for method in sol_method:
    qr = QROrthogonalDecomposition(A, b, sol_method=method)
    qr.fit_solve()
    print("正交化矩阵Q：\n", qr.Q)
    print("上三角矩阵R：\n", qr.R)
    print("线性方程组的解：\n", qr.x)
    print("验证解的精度：\n", qr.eps)
    print("=" * 60)
