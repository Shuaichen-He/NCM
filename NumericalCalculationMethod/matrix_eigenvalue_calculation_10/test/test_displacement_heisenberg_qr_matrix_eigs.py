# -*- coding: UTF-8 -*-
"""
@file_name: test_qr_matrix_eigs.py
@time: 2021-11-07
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from matrix_eigenvalue_calculation_10.displacement_heisenberg_qr_matrix_eigs import \
    DisplacementHeisenbergQRMatrixEig
from matrix_eigenvalue_calculation_10.qr_orthogonal_matrix_eigs import QROrthogonalMatrixEigenvalues



A = np.array([[2, 3, 4, 5, 6], [4, 4, 5, 6, 7], [0, 3, 6, 7, 8], [0, 0, 2, 8, 9], [0, 0, 0, 1, 0]])


methods = ["householder"]
for mt in methods:
    dhqr = DisplacementHeisenbergQRMatrixEig(A, eps=1e-15, transform=mt)
    dhqr.fit_eig()
    dhqr.plt_eigenvalues()
    print(mt, "：", dhqr.eigenvalues)
    print("=" * 60)

# hqr = DisplacementHeisenbergQRMatrixEig(A, eps=1e-15, transform="Givens", displacement="wilkins")
# hqr.fit_eig()
# hqr.plt_eigenvalues()
# qr = QROrthogonalMatrixEigenvalues(A, eps=1e-16, is_show=True)
# qr.fit_eig()
# qr.plt_eigenvalues()
# 验证
eig_vector = np.linalg.eig(A)
eig = eig_vector[0]  # 特征值
for i in range(len(eig)):
    print("%.15f" % eig[i])
# vector = eig_vector[1]  # 特征向量
# for e, vc in zip(eig, vector.T):  # 特征向量对应一列
#     print("%.20f" % e, end="")
#     max_v = np.max(np.abs(vc))
#     for v in vc:
#         print("%20.15f" % (v / max_v), end="")  #
#     print()
