# -*- coding: UTF-8 -*-
"""
@file_name: exp1.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from matrix_eigenvalue_calculation_10.utils.mat_eig_utils import MatrixEigenvalueUtils
from matrix_eigenvalue_calculation_10.power_method_eig import PowerMethodMatrixEig
from matrix_eigenvalue_calculation_10.origin_translation_inverse_power import OriginTranslationInversePower

A = np.array([[17, 0, 1, 0, 15], [23, 5, 7, 14, 16], [4, 0, 13, 0, 22], [10, 12, 19, 21, 3], [11, 18, 25, 2, 19]])
v0 = 0.1 * np.ones(5)

# 乘幂法
pmme = PowerMethodMatrixEig(A, v0=v0, eps=1e-16)
eig, vector = pmme.fit_eig()
utils = MatrixEigenvalueUtils(pmme.iter_eigenvalue, pmme.iter_eig_vector)
utils.show_iteration()
utils.plt_matrix_eig("乘幂法")
err = np.dot((A - eig * np.eye(5)), vector)
print(np.linalg.norm(err))

eig_vector = np.linalg.eig(A)
idx = np.argmax(abs(eig_vector[0]))  # 最大特征值索引
max_eig = eig_vector[0][idx]
max_vector = eig_vector[1][:, idx]
print(max_eig)
print(max_vector / max_vector[np.argmax(max_vector)])
print("=" * 80)

# 反幂法
pmme = PowerMethodMatrixEig(A, v0=v0, eps=1e-16, eig_type="inverse")
eig, vector = pmme.fit_eig()
utils = MatrixEigenvalueUtils(pmme.iter_eigenvalue, pmme.iter_eig_vector)
utils.show_iteration()
utils.plt_matrix_eig("反幂法")
err = np.dot((A - eig * np.eye(5)), vector)
print(np.linalg.norm(err))

eig_vector = np.linalg.eig(A)
idx = np.argmin(abs(eig_vector[0]))  # 最小特征值索引
min_eig = eig_vector[0][idx]
min_vector = eig_vector[1][:, idx]
print(min_eig)
print(min_vector / min_vector[np.argmax(min_vector)])
print("=" * 80)

# 原点平移反幂法
mius = [50, 18, 12, 2.2, -14]
for miu in mius:
    otip = OriginTranslationInversePower(A, u0=v0, miu=miu, eps=1e-16)
    otip.fit_eig()
    print("特征值：", otip.iter_eigenvalue[-1])
    print("特征向量", otip.eig_vector / np.max(np.abs(otip.eig_vector)))
eig_vector = np.linalg.eig(A)
for i in range(5):
    print("库函数所求特征值 %d：%.14f" % (i + 1, eig_vector[0][i]))
    print("特征向量：\n", eig_vector[1][:, i] / np.max(np.abs(eig_vector[1][:, i])))
print("=" * 80)

