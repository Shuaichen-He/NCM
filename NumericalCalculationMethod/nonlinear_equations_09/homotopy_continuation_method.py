# -*- coding: UTF-8 -*-
"""
@file_name: homotopy_continuation_method.py
@time: 2022-12-29
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from nonlinear_equations_09.jacobi_matrix import JacobiMatrix
from direct_solution_linear_equations_06.gaussian_elimination_algorithm import GaussianEliminationAlgorithm


class HomotopyContinuationMethod:
    """
    同伦延拓法，求解雅可比矩阵，给定网格划分N，采用N次龙格库塔法求解
    """

    def __init__(self, nlinear_Fxs, sym_vars, x0, N=4, method="newton"):
        self.sym_vars = sym_vars  # 定义的符号变量
        self.jacobi_obj = JacobiMatrix(nlinear_Fxs, sym_vars)  # 雅可比矩阵
        self.x0 = np.asarray(x0, dtype=np.float64).reshape(-1, 1)  # 迭代初始值，列向量形式
        self.N = N  # 子区间数
        self.method = method  # 分为newton和continuation
        self.roots, self.fxs_precision = None, None  # 近似解向量，以及解向量针对每个方程的精度

    @staticmethod
    def _solve_linear_equs(A, b):
        """
        采用高斯消元法求解线性方程组
        :param A: 系数矩阵
        :param b: 右端向量
        :return:
        """
        gea = GaussianEliminationAlgorithm(A, b)  # 默认列主元高斯消元法
        gea.fit_solve()  # 线性方程组求解
        return gea.x.reshape(-1, 1)

    def fit_roots(self):
        """
        核心算法：同伦延拓法，采用龙格库塔法求解微分方程组
        :return:
        """
        iter_, x_n = 0, self.x0  # 参数初始化
        jacobi_mat = self.jacobi_obj.solve_jacobi_mat()  # 求解雅可比矩阵
        h, b = 1 / self.N, self.jacobi_obj.cal_fx_values(self.x0)
        while iter_ < self.N:
            x_b = np.copy(x_n)  # 近似解的迭代
            if self.method.lower() == "newton":
                h = (iter_ + 1) / self.N
                b = self.jacobi_obj.cal_fx_values(x_b)  # 结合牛顿法
            A = self.jacobi_obj.cal_jacobi_mat_values(jacobi_mat, x_b)  # 雅可比矩阵值
            k1 = - h * self._solve_linear_equs(A, b)  # 列主元高斯消元法解线性方程组
            A = self.jacobi_obj.cal_jacobi_mat_values(jacobi_mat, x_b + k1 / 2)  # 雅可比矩阵值
            k2 = - h * self._solve_linear_equs(A, b)  # 解线性方程组
            A = self.jacobi_obj.cal_jacobi_mat_values(jacobi_mat, x_b + k2 / 2)  # 雅可比矩阵值
            k3 = - h * self._solve_linear_equs(A, b)  # 解线性方程组
            A = self.jacobi_obj.cal_jacobi_mat_values(jacobi_mat, x_b + k3)  # 雅可比矩阵值
            k4 = - h * self._solve_linear_equs(A, b)  # 解线性方程组
            x_n = x_b + (k1 + 2 * k2 + 2 * k3 + k4) / 6  # 近似解的更新
            iter_ = iter_ + 1  # 增1
        self.roots = x_n  # 满足精度的根
        # 最终解向量针对每个方程的精度
        self.fxs_precision = \
            self.jacobi_obj.cal_fx_values(self.roots.reshape(-1, 1)).flatten()
        return self.roots, self.fxs_precision