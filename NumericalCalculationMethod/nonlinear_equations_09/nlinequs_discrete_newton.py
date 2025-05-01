# -*- coding: UTF-8 -*-
"""
@file:nlinequs_discrete_newton.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from nonlinear_equations_09.utils.nonlinear_equations_utils import NonLinearEquationsUtils


class NLinearFxDiscreteNewton(NonLinearEquationsUtils):
    """
    离散牛顿迭代法求解非线性方程组，算法设计了下山因子，继承NonLinearEquationsUtils
    """

    def __init__(self, nlin_fxs, x0, h, eps=1e-10, max_iter=200, is_plt=False):
        NonLinearEquationsUtils.__init__(self, nlin_fxs, x0, max_iter, eps, is_plt)
        self.h = np.asarray(h, dtype=np.float64).reshape(-1, 1)  # 各分量的离散步长列向量
        self.n = len(x0)  # 解向量的个数
        self.fxs_precision = None  # 最终解向量针对每个方程的精度
        self.downhill_lambda = []  # 存储下山因子及其对应的迭代次数

    def diff_matrix(self, x_cur):
        """
        求解离散差商矩阵，参数x_cur为当前迭代的解向量
        :return:
        """
        disc_mat = np.zeros((self.n, self.n))  # 计算离散差商矩阵
        for i in range(self.n):  # 针对解向量每个变量xk
            x_d = np.copy(x_cur)  # 用于计算差商
            x_d[i] += self.h[i]  # 第i个变量xi：x_k + e_i * h_i
            disc_mat[:, i] = ((self.nlin_Fxs(x_d) - self.nlin_Fxs(x_cur)) /
                              self.h[i]).flatten()
        return disc_mat

    def fit_roots(self):
        """
        核心算法：离散牛顿法迭代法求解非线性方程组的解
        :return:
        """
        x_b = self.nlin_Fxs(self.x0)  # 标记上一次迭代解向量
        dx_n = self.diff_matrix(self.x0)  # 求解差商矩阵
        x_n = x_b - np.dot(np.linalg.inv(dx_n), x_b)  # 下一次迭代解向量
        sol_tol, iter_ = np.linalg.norm(self.nlin_Fxs(x_n)), 1  # 精度计算和迭代向量
        self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])
        while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
            x_b = np.copy(x_n)  # 解向量更新
            dx_n = self.diff_matrix(x_b)  # 求解差商矩阵
            if dx_n is True:  # 差商步长过小，满足精度要求
                break
            lambda_ = 1  # 下山因子
            sol_xb = self.nlin_Fxs(x_b)  # 上一次迭代的方程组的值向量
            # 是否保证在稳定下降收敛，以方程组值的范数进行判别标准
            while np.linalg.norm(self.nlin_Fxs(x_n)) > np.linalg.norm(sol_xb):
                lambda_ /= 2  # 逐次减半
                x_n = x_b - np.dot(lambda_ * np.linalg.inv(dx_n), sol_xb)  # 牛顿下山迭代公式
            if lambda_ < 1:  # 仅存储小于1的下山因子和当前迭代次数
                self.downhill_lambda.append([iter_, lambda_])
            else:
                x_n = x_b - np.dot(np.linalg.inv(dx_n), self.nlin_Fxs(x_b))  # 牛顿迭代公式
            sol_tol, iter_ = np.linalg.norm(x_n - x_b), iter_ + 1  # 更新精度和迭代变量
            self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])

        self.roots = self.iter_roots_precision[-1][1]  # 满足精度的根
        # 最终解向量针对每个方程的精度
        self.fxs_precision = self.nlin_Fxs(self.roots.reshape(-1, 1)).flatten()
        if self.is_plt:  # 是否可视化图像，参考不动点迭代法，略去...，修改参数title即可
            plt.figure(figsize=(14, 5))
            plt.subplot(121)
            self.plt_precision_convergence_curve(False, "Discrete \ Newton ")
            plt.subplot(122)
            self.plt_roots_convergence_curve(False, "Discrete \ Newton ")
            plt.show()
        return self.roots, self.fxs_precision
