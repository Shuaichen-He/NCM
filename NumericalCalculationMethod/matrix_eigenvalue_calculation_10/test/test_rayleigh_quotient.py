# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:test_chapoly.py
@time:2021/08/25
"""
from matrix_eigenvalue_calculation_10.rayleigh_quotient_accelerated_power import RayleighQuotientAcceleratedPower
from matrix_eigenvalue_calculation_10.power_method_eig import PowerMethodMatrixEig
import numpy as np
from matrix_eigenvalue_calculation_10.utils.mat_eig_utils import MatrixEigenvalueUtils

# A = np.array([[1, 5, 2], [5, -1, 7], [2, 7, 1]])  # 对称矩阵

# A = np.array([[1, 1, 0.5], [1, 1, 0.25], [0.5, 0.25, 2]])  # 非对称矩阵
# eig = np.linalg.eig(A)
# print("%.20f" % eig[0][1])
# print(eig[1])

# A = np.array([[3.6, 4.4, 0.8, -1.6, -2.8], [4.4, 2.6, 1.2, -0.4, 0.8], [0.8, 1.2, 0.8, -4.0, -2.8],
#               [-1.6, -0.4, -4.0, 1.2, 2.0], [-2.8, 0.8, -2.8, 2.0, 1.8]])
# u0 = np.ones(5)
# rqap = RayleighQuotientAcceleratedPower(A, u0, eps=1e-15)
# rqap.fit_eig()
# meu = MatrixEigenvalueUtils(rqap.iter_eigenvalue, rqap.iter_eig_vector)
# meu.show_iteration()
# meu.plt_matrix_eig("瑞利商")
# #
# power_eig = PowerMethodMatrixEig(A, u0, eps=1e-15, eig_type="power")
# eig, vector = power_eig.fit_eig()
# meu = MatrixEigenvalueUtils(power_eig.iter_eigenvalue, power_eig.iter_eig_vector)
# meu.show_iteration()
# meu.plt_matrix_eig("乘幂法")

# alpha, n = 0.25, 10
# a_diag = (1 + 2 * alpha) * np.ones(n)
# b_diag = -alpha * np.ones(n - 1)
# A = np.diag(a_diag) + np.diag(b_diag, k=1) + np.diag(b_diag, k=-1)
# np.random.seed(1)
# u0 = 0.1 * np.random.randn(n)

element = [1 / i for i in range(1, 12)]
A = np.zeros((6, 6))
for i in range(6):
    A[i, :] = element[i: 6 + i]
u0 = 0.5 * np.ones(6)
n = 6

rqap = RayleighQuotientAcceleratedPower(A, u0, eps=1e-16)
rqap.fit_eig()
meu = MatrixEigenvalueUtils(rqap.iter_eigenvalue, rqap.iter_eig_vector)
meu.show_iteration()
meu.plt_matrix_eig("瑞利商")
err = np.dot((A - rqap.eigenvalue * np.eye(n)), rqap.eig_vector)
print(np.linalg.norm(err))

eig_vector = np.linalg.eig(A)
idx = np.argmax(abs(eig_vector[0]))  # 最小特征值索引
min_eig = eig_vector[0][idx]
min_vector = eig_vector[1][:, idx]
print(min_eig)
print(min_vector / min_vector[np.argmax(min_vector)])


power_eig = PowerMethodMatrixEig(A, u0, eps=1e-16, eig_type="power")
eig, vector = power_eig.fit_eig()
meu = MatrixEigenvalueUtils(power_eig.iter_eigenvalue, power_eig.iter_eig_vector)
# meu.show_iteration()
meu.plt_matrix_eig("乘幂法")
err = np.dot((A - power_eig.eigenvalue * np.eye(n)), power_eig.eig_vector)
print(np.linalg.norm(err))
