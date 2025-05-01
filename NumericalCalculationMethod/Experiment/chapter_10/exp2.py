# -*- coding: UTF-8 -*-
"""
@file_name: exp2.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from matrix_eigenvalue_calculation_10.utils.mat_eig_utils import MatrixEigenvalueUtils
from matrix_eigenvalue_calculation_10.power_method_eig import PowerMethodMatrixEig
from matrix_eigenvalue_calculation_10.rayleigh_quotient_accelerated_power import RayleighQuotientAcceleratedPower
from matrix_eigenvalue_calculation_10.heisenberg_qr_matrix_eigs import HeisenbergQRMatrixEig
from matrix_eigenvalue_calculation_10.qr_orthogonal_matrix_eigs import QROrthogonalMatrixEigenvalues
from matrix_eigenvalue_calculation_10.displacement_heisenberg_qr_matrix_eigs import DisplacementHeisenbergQRMatrixEig

A = np.array([[1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6], [1 / 2, 1, 2 / 3, 1 / 2, 2 / 5, 1 / 3],
              [1 / 3, 2 / 3, 1, 3 / 4, 3 / 5, 1 / 2],
              [1 / 4, 1 / 2, 3 / 4, 1, 4 / 5, 2 / 3], [1 / 5, 2 / 5, 3 / 5, 4 / 5, 1, 5 / 6],
              [1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6, 1]])
v0 = 0.01 * np.ones(6)

# 瑞利商加速幂法
rqap = RayleighQuotientAcceleratedPower(A, u0=v0, eps=1e-16)
eig, eig_vector = rqap.fit_eig()
utils = MatrixEigenvalueUtils(rqap.iter_eigenvalue, rqap.iter_eig_vector)
utils.show_iteration()
utils.plt_matrix_eig("瑞利商加速幂法")
print("归一化后特征向量：", eig_vector)
err = np.dot((A - eig * np.eye(6)), eig_vector)
print(np.linalg.norm(err))

# 乘幂法与瑞利商加速幂法对比
# pmme = PowerMethodMatrixEig(A, v0=v0, eps=1e-16)
# eig, vector = pmme.fit_eig()
# utils = MatrixEigenvalueUtils(pmme.iter_eigenvalue, pmme.iter_eig_vector)
# utils.show_iteration()
# utils.plt_matrix_eig("乘幂法")
# err = np.dot((A - eig * np.eye(6)), vector)
# print(np.linalg.norm(err))

# 施密特正交化分解QR法求解矩阵的全部特征值
qrme = QROrthogonalMatrixEigenvalues(A, eps=1e-16)
qrme.fit_eig()
qrme.plt_eigenvalues()
print("施密特正交化全部特征值：", qrme.eigenvalues)
eig_vector = np.linalg.eig(A)
print("库函数：")
for i in range(6):
    print("%.16f" % eig_vector[0][i], end=", ")
print("\n", "=" * 80)

# 采用Schmidt、Givens、Householder三种QR正交分解法求解矩阵的全部特征值
methods = ["schmidt", "Givens", "householder"]
for mt in methods:
    hqr = HeisenbergQRMatrixEig(A, eps=1e-16, transform=mt)
    hqr.fit_eig()
    print(mt, hqr.eigenvalues)
    hqr.plt_eigenvalues()

# 采用位移Schmidt、位移Givens和位移Householder三种QR正交分解法求解矩阵的全部特征值.
methods = ["schmidt", "Givens", "householder"]
for mt in methods:
    dhqr = DisplacementHeisenbergQRMatrixEig(A, eps=1e-16, transform=mt)
    dhqr.fit_eig()
    print(mt, dhqr.eigenvalues)
    dhqr.plt_eigenvalues()
