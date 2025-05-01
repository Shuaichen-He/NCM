# -*- coding: UTF-8 -*-
"""
@file:nlinequs_newton.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
import matplotlib.pyplot as plt
from nonlinear_equations_09.jacobi_matrix import JacobiMatrix
from nonlinear_equations_09.utils.nonlinear_equations_utils import NonLinearEquationsUtils


class NLinearFxNewton(NonLinearEquationsUtils):
    """
    牛顿迭代法求解非线性方程组，包括牛顿法和下山牛顿法，继承NonLinearEquationsUtils
    """

    def __init__(self, nlinear_Fxs, sym_vars, x0, max_iter=200, eps=1e-10,
                 method="newton", is_plt=False):
        self.sym_vars = sym_vars  # 定义的符号变量
        nlin_equs_expr = sympy.lambdify([sym_vars], nlinear_Fxs, "numpy")  # 转换为数值方程
        NonLinearEquationsUtils.__init__(self, nlin_equs_expr, x0, max_iter,
                                         eps, is_plt)  # 父类初始化
        self.jacobi_obj = JacobiMatrix(nlinear_Fxs, sym_vars)  # 雅可比矩阵
        self.method = method  # 分为牛顿法和牛顿下山法
        self.fxs_precision = None  # 最终解向量针对每个方程的精度
        self.downhill_lambda = []  # 存储下山因子及其对应的迭代次数

    def fit_roots(self):
        """
        核心算法：牛顿迭代法求解非线性方程组的解
        :return:
        """
        x_b = self.jacobi_obj.cal_fx_values(self.x0)  # 方程组的函数值
        jacobi_mat = self.jacobi_obj.solve_jacobi_mat()  # 求解雅可比矩阵
        dx_n = self.jacobi_obj.cal_jacobi_mat_values(jacobi_mat, self.x0)  # 雅可比矩阵值
        x_n = x_b - np.dot(np.linalg.inv(dx_n), x_b)  # 第一次迭代值
        sol_tol = np.linalg.norm(self.jacobi_obj.cal_fx_values(x_n))  # 方程组解精度的范数作为精度控制
        iter_ = 1  # 迭代变量初始化
        self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])
        if self.method == "newton":
            self._solve_newton_(iter_, sol_tol, x_n, jacobi_mat)  # 牛顿法
        elif self.method == "downhill":
            self._solve_newton_downhill_(iter_, sol_tol, x_n, jacobi_mat)  # 牛顿下山法
        else:
            raise ValueError("仅支持方法：newton或downhill.")
        self.roots = self.iter_roots_precision[-1][1]  # 满足精度的根
        # 最终解向量针对每个方程的精度
        self.fxs_precision = self.jacobi_obj.cal_fx_values(self.roots.reshape(-1, 1)).flatten()
        if self.is_plt:  # 是否可视化图像
            plt.figure(figsize=(14, 5))
            plt.subplot(121)
            self.plt_precision_convergence_curve(False, "Newton \ " + self.method)
            plt.subplot(122)
            self.plt_roots_convergence_curve(False, "Newton \ " + self.method)
            plt.show()
        return self.roots, self.fxs_precision

    def _solve_newton_(self, iter_, sol_tol, x_n, jacobi_mat):
        """
        牛顿法求解非线性方程组的解
        :return:
        """
        while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
            x_b = np.copy(x_n)  # 解向量迭代
            dx_n = self.jacobi_obj.cal_jacobi_mat_values(jacobi_mat, x_b)  # 雅可比矩阵函数值
            x_n = x_b - np.dot(np.linalg.inv(dx_n), self.jacobi_obj.cal_fx_values(x_b))  # 牛顿法迭代公式
            # 以解向量x_n的方程值的范数作为精度判断
            sol_tol = np.linalg.norm(self.jacobi_obj.cal_fx_values(x_n))
            iter_ += 1  # 迭代次数增一
            self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])

    def _solve_newton_downhill_(self, iter_, sol_tol, x_n, jacobi_mat):
        """
        牛顿下山法求解非线性方程组的解
        :return:
        """
        while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
            x_b = np.copy(x_n)  # 解向量迭代
            dx_n = self.jacobi_obj.cal_jacobi_mat_values(jacobi_mat, x_b)  # 雅可比矩阵函数值
            x_n = x_b - np.dot(np.linalg.inv(dx_n), self.jacobi_obj.cal_fx_values(x_b))  # 牛顿法迭代公式
            lambda_ = 1  # 下山因子
            sol_xb = self.jacobi_obj.cal_fx_values(x_b)  # 上一次迭代的方程组的值向量
            # 是否保证在稳定下降收敛，以方程组值的范数进行判别标准
            while np.linalg.norm(self.jacobi_obj.cal_fx_values(x_n)) > \
                    np.linalg.norm(sol_xb):
                lambda_ /= 2  # 逐次减半
                x_n = x_b - np.dot(lambda_ * np.linalg.inv(dx_n), sol_xb)
            if lambda_ < 1:  # 仅存储小于1的下山因子和当前迭代次数
                self.downhill_lambda.append([iter_, lambda_])
            sol_tol, iter_ = np.linalg.norm(self.jacobi_obj.cal_fx_values(x_n)), iter_ + 1
            self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])
