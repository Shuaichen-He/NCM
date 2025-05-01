# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:test_chapoly.py
@time:2021/08/25
"""
from matrix_eigenvalue_calculation_10.power_method_eig import PowerMethodMatrixEig
import numpy as np
from matrix_eigenvalue_calculation_10.utils.mat_eig_utils import MatrixEigenvalueUtils

# A = np.array([[1, 5, 2], [6, -1, 7], [1, 3, 1]])
# A = np.array([[1, 2, -2], [-4, 3, 0], [5, 1, 4]])
# A = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 4]])
# A = np.array([[4, -1, 1], [-1, 3, -2], [1, -2, 3]])
# A = np.array([[1, 1, 0.5], [1, 1, 0.25], [0.5, 0.25, 2]])
# A = np.array([[5, 4, 1, 1], [4, 5, 1, 1], [1, 1, 4, 2], [1, 1, 2, 4]])

element = [1 / i for i in range(1, 12)]
A = np.zeros((6, 6))
for i in range(6):
    A[i, :] = element[i: 6 + i]
v0 = 0.5 * np.ones(6)
print(A)
# A = np.array([[2, 3, 4, 5, 6], [4, 4, 5, 6, 7], [0, 3, 6, 7, 8], [0, 0, 2, 8, 9], [0, 0, 0, 1, 0]])
# v0 = 0.5 * np.ones(5)

power_eig = PowerMethodMatrixEig(A, v0, eps=1e-16, max_iter=1000, eig_type="inverse")
eig, vector = power_eig.fit_eig()

meu = MatrixEigenvalueUtils(power_eig.iter_eigenvalue, power_eig.iter_eig_vector)
meu.show_iteration()
meu.plt_matrix_eig("乘幂法")
err = np.dot((A - eig * np.eye(6)), vector)
print(np.linalg.norm(err))

eig_vector = np.linalg.eig(A)
print(eig_vector)
idx = np.argmin(abs(eig_vector[0]))  # 最小特征值索引
min_eig = eig_vector[0][idx]
min_vector = eig_vector[1][:, idx]
print(min_eig)
print(min_vector / min_vector[np.argmax(min_vector)])
