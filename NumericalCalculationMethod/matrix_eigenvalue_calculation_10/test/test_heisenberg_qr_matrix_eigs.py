# -*- coding: UTF-8 -*-
"""
@file_name: test_qr_matrix_eigs.py
@time: 2021-11-07
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from matrix_eigenvalue_calculation_10.heisenberg_qr_matrix_eigs import HeisenbergQRMatrixEig
from matrix_eigenvalue_calculation_10.qr_orthogonal_matrix_eigs import QROrthogonalMatrixEigenvalues



# A = np.array([[3, 2, 1], [4, 3, 7], [5, 8, 6]])
# A = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 4]])
# A = np.array([[1, -2, 0], [-2, 2, -2], [0, -2, 3]])
# A = np.array([[1, 5, 6], [4, 7, 0], [8, 11, 4]])
# A = np.array([[-4, 1, 1, 1], [1, -4, 1, 1], [1, 1, -4, 1], [1, 1, 1, -4]])
# A = np.array([[2, 3, 4, 5, 6], [4, 4, 5, 6, 7], [0, 3, 6, 7, 8], [0, 0, 2, 8, 9], [0, 0, 0, 1, 0]])
# A = np.array([[5, -3, 2], [6, -4, 4], [4, -4, 5]])
# A = np.array([[6, 3, 2], [4, 3, 8], [7, 9, 5]])
A = np.array([[5, -1, 0, 0, 0], [-1, 4.5, 0.2, 0, 0], [0, 0.2, 1, -0.4, 0], [0, 0, -0.4, 3, 1], [0, 0, 0, 1, 3]])

methods = ["householder", "Givens"]  # householder,  Givens
for mt in methods:
    dhqr = HeisenbergQRMatrixEig(A, eps=1e-15, transform=mt)
    dhqr.fit_eig()
    dhqr.plt_eigenvalues()
    print(mt, "：", dhqr.eigenvalues)
    print("=" * 60)


# 验证
eig_vector = np.linalg.eig(A)
eig = eig_vector[0]  # 特征值
for i in range(len(eig)):
    print("%.15f" % eig[i])