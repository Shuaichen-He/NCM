# -*- coding: UTF-8 -*-
"""
@file_name: global_LM_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from nonlinear_equations_09.utils.nonlinear_equations_utils import NonLinearEquationsUtils
from direct_solution_linear_equations_06.doolittle_decomposition_lu \
    import DoolittleTriangularDecompositionLU  # 采用LU分解法求解方程组


class GlobalLevenbergMarquardt(NonLinearEquationsUtils):
    """
    全局化LM方法求解非线性方程组的解，以离散差商矩阵近似雅可比矩阵
    """

    def __init__(self, nlin_fxs, x0, h, delta=1.5, sigma=0.4, rho=0.5, eta=0.9,
                 max_iter=200, eps=1e-10, is_plt=False):
        NonLinearEquationsUtils.__init__(self, nlin_fxs, x0, max_iter, eps, is_plt)
        self.h = np.asarray(h, dtype=np.float64).reshape(-1, 1)  # 各分量的离散步长向量
        self.delta = delta  # 区间[1, 2]
        self.sigma, self.rho, self.eta = sigma, rho, eta  # 全局LM方法超参数
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
            disc_mat[:, i] = ((self.nlin_Fxs(x_d) - sol_xb) / self.h[i]).reshape(-1)
        return disc_mat

    def fit_nlinequs_roots(self):
        """
        全局化LM方法，求解非线性方程组的解，核心算法
        :return:
        """
        iter_, sol_tol, x_n = 0, np.infty, self.x0  # 参数初始化
        miu_k = np.linalg.norm(self.nlin_Fxs(self.x0)) ** self.delta  # ||F(x)||^delta
        while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
            x_b = np.copy(x_n)  # 解向量的更新
            sol_xb = self.nlin_Fxs(x_b)  # F(x_k)
            diff_mat = self.diff_mat(x_b, sol_xb)  # 差商矩阵代替雅可比矩阵
            g_k = np.dot(diff_mat.T, sol_xb)  # 价值函数的梯度
            if np.linalg.norm(g_k) < self.eps:  # 停机规则
                break
            # 采用LU分解法求解方程组，计算搜索方向
            A = -(np.dot(diff_mat.T, diff_mat) + miu_k * np.eye(self.n))
            dtd = DoolittleTriangularDecompositionLU(A, g_k.reshape(-1),
                                                     sol_method="pivot")
            s_k = dtd.fit_solve().reshape(-1, 1)  # 解
            if np.linalg.norm(self.nlin_Fxs(x_b + s_k)) <= \
                    self.eta * np.linalg.norm(sol_xb):
                x_n = x_b + s_k  # 第（3）步
            else:
                m, alpha_k = 0, 1  # 在m < 20次内确定参数alpha_k
                while m < 20:  # 采用Armijo线搜索求步长，循环搜索20次
                    term_1 = self._value_fun(x_b + self.sigma * self.rho ** m * s_k)
                    term_2 = self._value_fun(x_b) + \
                             self.sigma * self.rho ** m * np.dot(g_k.T, s_k)
                    if term_1 <= term_2:
                        alpha_k = self.sigma * self.rho ** m
                        break
                    m += 1
                print("alpha_k", alpha_k)
                x_n = x_b + alpha_k * s_k  # 更新下一次迭代解向量
            miu_k = np.linalg.norm(self.nlin_Fxs(x_n)) ** self.delta  # ||F(x)||^delta，迭代
            sol_tol, iter_ = np.linalg.norm(x_n - x_b), iter_ + 1  # 更新精度和迭代次数
            self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])
            if np.linalg.norm(self.nlin_Fxs(x_n)) <= self.eps:  # 新解满足精度要求，则停机
                break
        self.roots = self.iter_roots_precision[-1][1]  # 满足精度的根
        # 最终解向量针对每个方程的精度
        self.fxs_precision = self.nlin_Fxs(self.roots.reshape(-1, 1)).flatten()
        if self.is_plt:  # 是否可视化图像
            plt.figure(figsize=(14, 5))
            plt.subplot(121)
            self.plt_precision_convergence_curve(False, "Global \ LM")
            plt.subplot(122)
            self.plt_roots_convergence_curve(False, "Global \ LM")
            plt.show()
        return self.roots, self.fxs_precision

    def _value_fun(self, x):
        """
        计算价值函数
        :return:
        """
        F_x = self.nlin_Fxs(x)  # F(x)
        return 0.5 * np.dot(F_x.T, F_x)
