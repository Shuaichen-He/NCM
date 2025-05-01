# -*- coding: UTF-8 -*-
"""
@file_name: test_oigtran_inverse_power.py
@time: 2021-11-06
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from matrix_eigenvalue_calculation_10.origin_translation_inverse_power import OriginTranslationInversePower
from matrix_eigenvalue_calculation_10.utils.mat_eig_utils import MatrixEigenvalueUtils

A = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 4]])
# A = np.array([[1, 2, -2], [-4, 3, 0], [5, 1, 4]])
u0 = np.ones(3)
miu = [4, 2.8, 1.1]
for m in miu:
    oig_trans = OriginTranslationInversePower(A, u0, miu=m, eps=1e-15)
    oig_trans.fit_eig()
    # meu = MatrixEigenvalueUtils(oig_trans.iter_eigenvalue, oig_trans.iter_eig_vector)
    # meu.show_iteration()
    # meu.plt_matrix_eig("原点平移法$\lambda=%.2f$" % m)
    print(oig_trans.eig_vector)

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
