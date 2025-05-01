# -*- coding: UTF-8 -*-
"""
@file:damped_least_square_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from nonlinear_equations_09.utils.nonlinear_equations_utils import NonLinearEquationsUtils
from direct_solution_linear_equations_06.doolittle_decomposition_lu \
    import DoolittleTriangularDecompositionLU  # 采用LU分解法求解方程组


class DampedLeastSquare_LM(NonLinearEquationsUtils):
    """
    阻尼最小二乘算法，即Levenberg-Marquarat算法，是Gauss-Newton算法的一种修正法
    """

    def __init__(self, nlin_fxs, x0, h, u, v, max_iter=200, eps=1e-10, is_plt=False):
        NonLinearEquationsUtils.__init__(self, nlin_fxs, x0, max_iter, eps, is_plt)
        self.h = np.asarray(h, dtype=np.float64).reshape(-1, 1)  # 各分量的离散步长向量
        self.u = u  # 阻尼因子
        self.v = v  # 缩放常数
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
        阻尼最小二乘法求解非线性方程组的解，核心算法
        :return:
        """
        iter_, sol_tol, x_n = 0, np.infty, self.x0  # 必要参数的初始化
        while_flag = True  # norm(pk) < eps
        while while_flag and np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
            x_b = np.copy(x_n)  # 解向量的更新
            sol_xb = self.nlin_Fxs(x_b)  # F(x_k)
            diff_mat = self.diff_mat(x_b, sol_xb)  # 求解离散差商矩阵
            d_fai = np.dot(diff_mat.T, sol_xb)  # 1/2F'F一阶导的值
            fal_val = np.dot(sol_xb.T, sol_xb) / 2  # 对应dφ(x)的值
            flag, descent_j = 0, 0  # 计算阻尼因子和x_n的判断标记
            while flag == 0:
                A = np.dot(diff_mat.T, diff_mat) + self.u * np.eye(self.n)
                dtd = DoolittleTriangularDecompositionLU(A, d_fai.reshape(-1), sol_method="pivot")
                sol = dtd.fit_solve()  # 采用LU分解法求解方程组
                # sol = np.linalg.solve(A, d_fai)  # 库函数求解
                if np.linalg.norm(sol) < self.eps:  # 解向量的范数小于精度要求
                    while_flag = False
                    break
                x_n = x_b - sol.reshape(-1, 1)  # 下一次迭代解向量x_{k+1}
                sol_xn = self.nlin_Fxs(x_n)  # F(x_{k+1})
                fal_val_n = np.dot(sol_xn.T, sol_xn) / 2  # fai(x)
                if fal_val_n < fal_val:  # 第（5）步
                    if descent_j == 0:
                        self.u, descent_j = self.u / self.v, 1
                    else:
                        flag = 1
                else:  # 第（6）步
                    self.u, descent_j = self.u * self.v, 1
                    if np.linalg.norm(x_n - x_b) < self.eps:
                        flag = 1
            sol_tol, iter_ = np.linalg.norm(d_fai), iter_ + 1  # 更新精度和迭代次数
            self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])

        self.roots = self.iter_roots_precision[-1][1]  # 满足精度的根
        # 最终解向量针对每个方程的精度
        self.fxs_precision = self.nlin_Fxs(self.roots.reshape(-1, 1)).flatten()
        if self.is_plt:  # 是否可视化图像
            plt.figure(figsize=(14, 5))
            plt.subplot(121)
            self.plt_precision_convergence_curve(False, "Damped \ LS")
            plt.subplot(122)
            self.plt_roots_convergence_curve(False, "Damped \ LS")
            plt.show()
        return self.roots, self.fxs_precision

