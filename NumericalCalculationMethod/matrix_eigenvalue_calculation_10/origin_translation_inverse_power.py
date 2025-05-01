# -*- coding: UTF-8 -*-
"""
@file_name: origin_translation_inverse_power.py
@time: 2021-11-06
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from direct_solution_linear_equations_06.gaussian_elimination_algorithm \
    import GaussianEliminationAlgorithm  # 导入高斯消元类


class OriginTranslationInversePower:
    """
    原点平移反幂法求矩阵离某个特定的常数最近的特征值及其对用的特征向量的迭代法
    """

    def __init__(self, A, u0, miu, max_iter=1000, eps=1e-8):
        self.A = np.asarray(A, dtype=np.float64)  # 待求矩阵
        if np.linalg.norm(u0) <= 1e-15:
            raise ValueError("初始向量不能为零向量或初始向量值过小！")
        else:
            self.u0 = np.asarray(u0, dtype=np.float64)
        self.miu = miu  # 离某个特征值最近的常数设置
        self.eps, self.max_iter = eps, max_iter  # 精度要求和最大迭代次数
        self.eigenvalue = -np.infty  # 特征值
        self.eig_vector = None  # 特征向量
        self.iter_eigenvalue = []  # 迭代过程的特征值的变化
        self.iter_eig_vector = []  # 迭代过程中特征向量的变化

    def fit_eig(self):
        """
        核心算法：原点平移反幂法迭代求解
        :return:
        """
        self.eig_vector = self.u0  # 按模最小特征值对应的特征向量
        self.eigenvalue, max_scalar = 0, 0  # 按模最小特征值
        tol, iter_ = np.infty, 0  # 初始精度和迭代次数
        while np.abs(tol) > self.eps and iter_ < self.max_iter:
            eig_value = max_scalar
            # 如下采用高斯列主元消元法，可实验其他方法
            A = self.A - self.miu * np.eye(len(self.u0))
            guass_eliminat = GaussianEliminationAlgorithm(A, self.eig_vector)
            guass_eliminat.fit_solve()
            v_k = np.copy(guass_eliminat.x)
            max_scalar = np.max(v_k)  # max_scalar为按模最大的标量
            self.iter_eigenvalue.append([iter_, 1 / max_scalar + self.miu])
            self.eig_vector = np.copy(v_k / max_scalar)  # 归一化
            self.iter_eig_vector.append(self.eig_vector)  # 存储归一化后的
            iter_, tol = iter_ + 1, abs(max_scalar - self.eigenvalue)
            self.eigenvalue = max_scalar
        self.eigenvalue = 1 / max_scalar + self.miu
        return self.eigenvalue, self.eig_vector
