# -*- coding: UTF-8 -*-
"""
@file_name: test_qr_matrix_eigs.py
@time: 2021-11-07
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from matrix_eigenvalue_calculation_10.qr_orthogonal_matrix_eigs import QROrthogonalMatrixEigenvalues


A = np.array([[3, 2, 1], [4, 3, 7], [5, 8, 6]])
# A = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 4]])
# A = np.array([[1, -2, 0], [-2, 2, -2], [0, -2, 3]])
# A = np.array([[1, 5, 6], [4, 7, 0], [8, 11, 4]])
# A = np.array([[4, 1, -2, 2], [1, 2, 0, 1], [-2, 0, 3, -2], [2, 1, -2, -1]])
# A = np.array([[5, -1, 0, 0, 0], [-1, 4.5, 0.2, 0, 0], [0, 0.2, 1, -0.4, 0], [0, 0, -0.4, 3, 1], [0, 0, 0, 1, 3]])
# A = np.array([[2, 3, 4, 5, 6], [4, 4, 5, 6, 7], [0, 3, 6, 7, 8], [0, 0, 2, 8, 9], [0, 0, 0, 1, 0]])
qr = QROrthogonalMatrixEigenvalues(A, eps=1e-15, is_show=True)
qr.fit_eig()
qr.plt_eigenvalues()
for e in qr.eigenvalues:
    print("%.20f" % e)
# 验证
eig_vector = np.linalg.eig(A)
eig = eig_vector[0]  # 特征值
vector = eig_vector[1]  # 特征向量
for e, vc in zip(eig, vector.T):  # 特征向量对应一列
    print("%.20f" % e, end="")
    max_v = np.max(np.abs(vc))
    for v in vc:
        print("%20.15f" % (v / max_v), end="")  #
    print()

