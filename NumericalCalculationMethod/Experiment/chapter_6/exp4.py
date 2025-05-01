# -*- coding: UTF-8 -*-
"""
@file_name: exp4.py
@time: 2023-02-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from direct_solution_linear_equations_06.qr_orthogonal_decomposition import QROrthogonalDecomposition


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
