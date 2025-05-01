# -*- coding: UTF-8 -*-
"""
@file:nlinequs_newton_sor.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from nonlinear_equations_09.utils.nonlinear_equations_utils import NonLinearEquationsUtils


class NLinearFxNewtonSOR(NonLinearEquationsUtils):
    """
    牛顿—SOR类迭代法求解非线性方程组， 继承NonLinearEquationsUtils
    """

    def __init__(self, nlin_fxs, x0, h, jacobi_para=3, sor_factor=1.0,
                 max_iter=200, eps=1e-10, method="jacobi", is_plt=False):
        NonLinearEquationsUtils.__init__(self, nlin_fxs, x0, max_iter, eps, is_plt)
        self.h = np.asarray(h, dtype=np.float64).reshape(-1, 1)  # 各分量的离散步长向量
        self.jp = jacobi_para  # 雅可比迭代的参量
        self.w = sor_factor  # SOR法的松弛因子
        self.n = len(x0)  # 解向量的个数
        self.method = method  # 雅可比法jacobi和松弛法sor
        self.fxs_precision = None  # 最终解向量针对每个方程的精度

    def diff_matrix(self, x_cur):
        """
        求解差商矩阵，x_cur为当前迭代的解向量
        :return:
        """
        disc_mat = np.zeros((self.n, self.n))  # 计算差商离散矩阵
        for i in range(self.n):  # 针对解向量每个变量xk
            x_d = np.copy(x_cur)  # 用于计算差商
            x_d[i] += self.h[i]  # 第i个变量xi：x_k + e_i * h_i
            disc_mat[:, i] = ((self.nlin_Fxs(x_d) - self.nlin_Fxs(x_cur)) / self.h[i]).flatten()
        return disc_mat

    def fit_roots(self):
        """
        核心算法：牛顿—SOR类迭代法求解非线性方程组的解
        :return:
        """
        iter_, sol_tol, x_n = 0, np.infty, self.x0  # 初始化迭代变量、精度和迭代解向量
        if self.method.lower() == "jacobi":  # 牛顿—雅可比迭代法
            while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
                x_b = np.copy(x_n)  # 解向量的迭代
                Ak = self.diff_matrix(x_b)  # 离散法求得雅可比矩阵
                D = np.diag(np.diag(Ak))  # 对角矩阵
                if np.max(np.abs(np.diag(Ak))) < self.eps:  # 对角元素绝对值最大者较小
                    D = D + 0.05 * np.eye(self.n)  # 对角矩阵添加0.05*I，保证可逆
                H = np.linalg.inv(D) * (Ak - D)
                H_m = np.eye(self.n)
                for i in range(1, self.jp - 1):
                    H_m = H_m + np.power(H, i)  # 矩阵的幂次累加
                x_n = x_b - np.dot(np.dot(H_m, np.linalg.inv(D)), self.nlin_Fxs(x_b))
                sol_tol, iter_ = np.linalg.norm(x_n - x_b), iter_ + 1  # 更新精度和迭代变量
                self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])
        elif self.method.lower() == "sor":  # 牛顿—SOR法
            while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
                x_b = np.copy(x_n)  # 解向量的迭代
                Ak = self.diff_matrix(x_b)  # 离散法求得雅可比矩阵
                D = np.diag(np.diag(Ak))  # 对角矩阵
                L, U = - np.tril(Ak - D), - np.triu(Ak - D)  # 下三角、上三角矩阵
                H1_inv = np.linalg.inv(D - self.w * L)
                H = np.dot(H1_inv, (1 - self.w) * D + self.w * U)
                H_m = np.eye(self.n)
                for i in range(1, self.jp - 1):
                    H_m = H_m + np.power(H, i)
                x_n = x_b - np.dot(self.w * np.dot(H_m, H1_inv), self.nlin_Fxs(x_b))
                sol_tol, iter_ = np.linalg.norm(x_n - x_b), iter_ + 1  # 更新精度和迭代变量
                self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])
        else:
            raise ValueError("仅支持牛顿—Jacobi法和牛顿—SOR法。")
        self.roots = self.iter_roots_precision[-1][1]  # 满足精度的根
        # 最终解向量针对每个方程的精度
        self.fxs_precision = self.nlin_Fxs(self.roots.reshape(-1, 1)).flatten()
        if self.is_plt:  # 是否可视化图像
            plt.figure(figsize=(14, 5))
            plt.subplot(121)
            self.plt_precision_convergence_curve(False, " newton \ %s" % self.method.upper())
            plt.subplot(122)
            self.plt_roots_convergence_curve(False, "newton \ %s" % self.method.upper())
            plt.show()
        return self.roots, self.fxs_precision
