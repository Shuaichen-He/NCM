# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:test_chapoly.py
@time:2021/08/25
"""
from matrix_eigenvalue_calculation_10.shrinkage_method_matrix_all_eig import ShrinkageMatrixAllEigenvalues
import numpy as np


if __name__ == '__main__':
    # A = np.array([[3, 2, 1], [4, 3, 7], [5, 8, 6]])
    # A = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 4]])
    # A = np.array([[1, 2, 1, -3], [2, 5, 0, -5], [1, 0, 14, 1], [-3, -5, 1, 15]])
    A = np.array([[-4, 1, 1, 1], [1, -4, 1, 1], [1, 1, -4, 1], [1, 1, 1, -4]])
    # A = np.array([[1, 5, 2], [5, -1, 7], [2, 7, 1]])
    # A = np.array([[2, 3, 4, 5, 6], [4, 4, 5, 6, 7], [0, 3, 6, 7, 8], [0, 0, 2, 8, 9], [0, 0, 0, 1, 0]])
    sm = ShrinkageMatrixAllEigenvalues(A, eps=1e-15)
    eigs = sm.fit_eig()
    for i in range(len(eigs)):
        print("%.20f" % eigs[i])
    eig_vector = np.linalg.eig(A)
    print(eig_vector[0])
    # eig = eig_vector[0]  # 特征值
    # vector = eig_vector[1]  # 特征向量
    # for e, vc in zip(eig, vector.T):  # 特征向量对应一列
    #     print("%.20f" % e, end="")
    #     max_v = np.max(np.abs(vc))
    #     for v in vc:
    #         print("%20.15f" % (v / max_v), end="")  #
    #     print()