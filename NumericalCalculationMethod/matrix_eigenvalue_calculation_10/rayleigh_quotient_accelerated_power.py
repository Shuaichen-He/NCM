# -*- coding: UTF-8 -*-
"""
@file:rayleigh_quotient_accelerated_power.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import decimal
decimal.getcontext().prec = 128


class RayleighQuotientAcceleratedPower:
    """
    瑞利商加速幂法，对称矩阵可加快幂法的收敛速度，非对称矩阵，可能加速效果不明显。
    """

    def __init__(self, A, u0, max_iter=1000, eps=1e-8):
        self.A = np.asarray(A, dtype=np.float64)  # 待求矩阵
        self.is_matrix_symmetric()  # 判断是否为对称矩阵
        if np.linalg.norm(u0) <= 1e-15:
            raise ValueError("初始向量不能为零向量或初始向量值过小！")
        else:
            self.u0 = np.asarray(u0, dtype=np.float64)
        self.eps, self.max_iter = eps, max_iter  # 精度要求和最大迭代次数
        self.eigenvalue = -np.infty  # 主特征值
        self.eig_vector = None  # 主特征向量
        self.iter_eigenvalue = []  # 迭代过程的主特征值的变化
        self.iter_eig_vector = []  # 迭代过程中主特征向量的变化

    def is_matrix_symmetric(self):
        """
        判断矩阵是否是对称矩阵，非对称矩阵，加速效果不明显
        :return:
        """
        X = np.triu(self.A)  # 取矩阵上三角
        X += X.T - np.diag(X.diagonal())  # 构造对称矩阵
        if (self.A == X).all():
            print("对称矩阵，可用瑞利商加速幂法加速收敛速度！")
        else:
            print("非对称矩阵，瑞利商加速幂法收敛速度可能不明显！")

    def fit_eig(self):
        """
        瑞利商加速幂法求解矩阵主特征值和主特征向量
        :return: 主特征值eigenvalue和对应的特征向量eig_vector
        """
        self.eig_vector, self.eigenvalue = np.copy(self.u0), np.infty  # 主特征向量和主特征值
        tol, iter_ = np.infty, 0  # 初始精度和迭代次数
        while np.abs(tol) > self.eps and iter_ < self.max_iter:
            # vector_old = np.copy(self.eig_vector)  # 用于精度判断
            v_k = np.dot(self.A, self.eig_vector)
            rayleigh_q = np.dot(v_k.T, self.eig_vector) / \
                         np.dot(self.eig_vector.T, self.eig_vector)  # 瑞利商
            self.iter_eigenvalue.append([iter_, rayleigh_q])
            self.eig_vector = v_k / rayleigh_q
            # tol, iter_ = np.linalg.norm(self.eig_vector - vector_old), iter_ + 1  # 特征向量的精度
            self.iter_eig_vector.append(self.eig_vector)  # 存储的特征向量未进行归一化
            tol, iter_ = np.abs(rayleigh_q - self.eigenvalue), iter_ + 1  # 更新精度和迭代变量
            self.eigenvalue = rayleigh_q
        self.eig_vector = self.eig_vector / np.max(np.abs(self.eig_vector))  # 归一化
        return self.eigenvalue, self.eig_vector
