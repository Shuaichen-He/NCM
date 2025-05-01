# -*- coding: UTF-8 -*-
"""
@file:shrinkage_method_matrix_all_eig.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np


class ShrinkageMatrixAllEigenvalues:
    """
    收缩法求解矩阵全部的特征值，基于幂法求解特征值的方法
    如果矩阵是病态矩阵或矩阵有复特征值，出现错误结果。
    """

    def __init__(self, A, eps=1e-8):
        self.A = np.asarray(A, dtype=np.float64)
        self.n = self.A.shape[0]
        self.eps = eps  # 精度要求
        self.eigenvalues = np.zeros(self.n)  # 存储矩阵全部特征值

    def fit_eig(self):
        """
        利用收缩法求解矩阵全部特征值
        :return:
        """
        shrink_B = np.copy(self.A)  # 收缩的矩阵B
        for i in range(self.n):
            # 幂法求解主特征值和主特征向量
            self.eigenvalues[i], u = \
                self._power_method(shrink_B, 0.1 * np.ones(shrink_B.shape[0]))
            u /= u[0]  # 主特征向量归一化
            mat_b = shrink_B - u.reshape(-1, 1) * shrink_B[0, :]
            shrink_B = mat_b[1:shrink_B.shape[0], 1:shrink_B.shape[0]]  # B为收缩后的矩阵
        self.eigenvalues = sorted(self.eigenvalues, reverse=True)  # 降序排列
        return self.eigenvalues

    def _power_method(self, A, x0, max_iter=1000):
        """
        幂法求解矩阵的主特征值和主特征向量
        :param A: 不断压缩的矩阵
        :param x0: 迭代初值
        :param max_iter: 最大迭代次数
        :return: 主特征值和主特征向量
        """
        max_eig_vector, max_eigenvalue = x0, np.infty  # 主特征向量,主特征值
        tol, iter_ = np.infty, 0  # 初始精度和迭代次数
        while np.abs(tol) > self.eps and iter_ < max_iter:
            y = np.dot(A, max_eig_vector)
            max_scalar = np.max(y)  # max_scalar为按模最大的标量
            max_eig_vector = y / max_scalar
            iter_, tol = iter_ + 1, np.abs(max_scalar - max_eigenvalue)  # 更新迭代变量和精度
            max_eigenvalue = max_scalar
        if iter_ == max_iter:
            print("幂法求解特征值“%f”可能不收敛." % max_eigenvalue)
        else:
            print("幂法求解特征值“%f”，迭代次数%d." % (max_eigenvalue, iter_))
        return max_eigenvalue, max_eig_vector
