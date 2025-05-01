# -*- coding: UTF-8 -*-
"""
@file:gauss_newton.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from nonlinear_equations_09.utils.nonlinear_equations_utils import NonLinearEquationsUtils


class GaussNewtonIteration(NonLinearEquationsUtils):
    """
    高斯牛顿迭代法求解非线性方程组，继承NonLinearEquationsUtils
    """

    def __init__(self, nlin_fxs, x0, h, max_iter=200, eps=1e-10, is_plt=False):
        NonLinearEquationsUtils.__init__(self, nlin_fxs, x0, max_iter, eps, is_plt)
        self.h = np.asarray(h, dtype=np.float64).reshape(-1, 1)  # 各分量的离散步长向量
        self.n = len(x0)  # 解向量的个数
        self.fxs_precision = None  # 最终解向量针对每个方程的精度

    def diff_mat(self, x_b, sol_xb):
        """
        求解差商矩阵
        :return:
        """
        disc_mat = np.zeros((self.n, self.n))  # 计算差商离散矩阵
        for i in range(self.n):
            x_d = np.copy(x_b)
            x_d[i] += self.h[i]
            disc_mat[:, i] = ((self.nlin_Fxs(x_d) - sol_xb) / self.h[i]).flatten()
        return disc_mat

    def fit_roots(self):
        """
        核心算法：高斯牛顿法迭代法求解非线性方程组的解
        :return:
        """
        iter_, sol_tol, x_n = 0, np.infty, self.x0  # 初始化参数
        while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
            x_b = np.copy(x_n)  # 解向量的更新
            sol_xb = self.nlin_Fxs(x_b)  # F(x_k)
            diff_mat = self.diff_mat(x_b, sol_xb)  # 离散差商矩阵
            # 判断Hessian矩阵np.dot(diff_mat.T, diff_mat)是否满秩
            hessian = np.dot(diff_mat.T, diff_mat)
            if np.linalg.matrix_rank(hessian) == hessian.shape[1]:
                delta_F = np.linalg.inv(hessian)  # Jk^T * Jk
                x_n = x_b - np.dot(np.dot(delta_F, diff_mat.T), sol_xb)  # 高斯—牛顿法公式
                iter_, sol_tol = iter_ + 1, np.linalg.norm(x_n - x_b)
                self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])
            else:
                raise ValueError("非列满秩，求解失败.")

        self.roots = self.iter_roots_precision[-1][1]  # 满足精度的根
        # 最终解向量针对每个方程的精度
        self.fxs_precision = self.nlin_Fxs(self.roots.reshape(-1, 1)).flatten()
        if self.is_plt:  # 是否可视化图像
            plt.figure(figsize=(14, 5))
            plt.subplot(121)
            self.plt_precision_convergence_curve(False, "Gauss-Newton")
            plt.subplot(122)
            self.plt_roots_convergence_curve(False, "Gauss-Newton")
            plt.show()
        return self.roots, self.fxs_precision
