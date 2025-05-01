# -*- coding: UTF-8 -*-
"""
@file:power_method_eig.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from direct_solution_linear_equations_06.gaussian_elimination_algorithm \
    import GaussianEliminationAlgorithm  # 导入高斯消元法类


class PowerMethodMatrixEig:
    """
    幂法求解矩阵按模最大特征值和对应特征向量，以及反幂法求解矩阵的按模最小特征值和对应特征向量
    """

    def __init__(self, A, v0, max_iter=1000, eps=1e-8, eig_type="power"):
        self.A = np.asarray(A, dtype=np.float64)
        print("矩阵的条件数为：", np.linalg.cond(A))
        if np.linalg.norm(v0) <= 1e-15:
            raise ValueError("初始向量不能为零向量或初始向量值过小！")
        self.v0 = np.asarray(v0, dtype=np.float64)
        self.eps, self.max_iter = eps, max_iter  # 精度要求和最大迭代次数
        self.eig_type = eig_type  # 按模最大power、按模最小inverse
        self.eigenvalue = 0  # 主（按模最小）特征值
        self.eig_vector = None  # 主（按模最小）特征向量
        self.iter_eigenvalue = []  # 迭代过程的主（按模最小）特征值的变化
        self.iter_eig_vector = []  # 迭代过程中主（按模最小）特征向量的变化

    def fit_eig(self):
        """
        幂法求解矩阵按模最大和按模最小特征值和对应的特征向量
        :return:
        """
        if self.eig_type == "power":  # 乘幂法
            return self._fit_power_()
        elif self.eig_type == "inverse":  # 反幂法
            return self._fit_inverse_power_()
        else:
            raise ValueError("eig_type参数仅能为power或inverse")

    def _fit_power_(self):
        """
        核心算法：幂法求解矩阵主特征值和主特征向量
        :return: 主特征值eigenvalue和对应的特征向量eig_vector
        """
        self.eig_vector, self.eigenvalue = self.v0, np.infty  # 初始化主特征向量和主特征值
        tol, iter_ = np.infty, 0  # 初始精度和迭代次数
        while np.abs(tol) > self.eps and iter_ < self.max_iter:
            vk = np.dot(self.A, self.eig_vector)
            max_scalar = np.max(vk)  # max_scalar为按模最大的标量
            self.iter_eigenvalue.append([iter_, max_scalar])
            self.eig_vector = vk / max_scalar  # 归一化
            self.iter_eig_vector.append(self.eig_vector)
            iter_, tol = iter_ + 1, np.abs(max_scalar - self.eigenvalue)  # 更新迭代变量和精度
            self.eigenvalue = max_scalar  # 更新，max_scalar既用于归一化，又用于精度判断
        return self.eigenvalue, self.eig_vector

    def _fit_inverse_power_(self):
        """
        核心算法：幂法求解矩阵按模最小特征值和对应的特征向量
        :return: 按模最小特征值eigenvalue和对应的特征向量eig_vector
        """
        self.eig_vector, self.eigenvalue = self.v0, 0  # 按模最小特征值对应的特征向量
        tol, iter_ = np.infty, 0  # 初始精度和迭代次数
        while np.abs(tol) > self.eps and iter_ < self.max_iter:
            # 如下采用高斯列主元消元法，可实验其他方法
            gea = GaussianEliminationAlgorithm(self.A, self.eig_vector)
            gea.fit_solve()  # 高斯列主元消元法
            v_k = np.copy(gea.x)
            max_scalar = np.max(v_k)  # max_scalar为按模最大的标量
            self.iter_eigenvalue.append([iter_, 1 / max_scalar])
            self.eig_vector = v_k / max_scalar  # 归一化
            self.iter_eig_vector.append(self.eig_vector)
            iter_, tol = iter_ + 1, np.abs(max_scalar - self.eigenvalue)  # 更新迭代变量和精度
            self.eigenvalue = max_scalar  # 更新，max_scalar既用于归一化，又用于精度判断
        self.eigenvalue = 1 / self.eigenvalue  # 取倒数
        return self.eigenvalue, self.eig_vector