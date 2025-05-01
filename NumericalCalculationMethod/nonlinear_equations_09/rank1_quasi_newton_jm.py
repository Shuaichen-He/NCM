# -*- coding: UTF-8 -*-
"""
@file:rank1_quasi_newton_jm.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
import matplotlib.pyplot as plt
from nonlinear_equations_09.utils.nonlinear_equations_utils import NonLinearEquationsUtils
from nonlinear_equations_09.jacobi_matrix import JacobiMatrix


class Rank1QuasiNewton(NonLinearEquationsUtils):
    """
    秩1算法求解非线性方程组的解，包括Broyden算法、Broyden第二方法、逆Broyden算法和逆Broyden第二方法
    继承NonLinearEquationsUtils, 采用雅可比矩阵作为A0
    """

    def __init__(self, nlin_fxs, sym_vars, x0, max_iter=200, eps=1e-10,
                 method="broyden", is_plt=False):
        # 符号非线性方程组转化为lambda函数
        nlin_equs_expr = sympy.lambdify([sym_vars], nlin_fxs, "numpy")
        NonLinearEquationsUtils.__init__(self, nlin_equs_expr, x0, max_iter,
                                         eps, is_plt)
        self.n = len(x0)  # 解向量的个数
        self.method = method  # 秩1四种算法
        self.jacobi_obj = JacobiMatrix(nlin_fxs, sym_vars)  # 雅可比矩阵
        self.fxs_precision = None  # 最终解向量针对每个方程的精度

    def fit_nlin_roots(self):
        """
        核心算法: 秩1迭代法求解非线性方程组的解，根据方法选择对应的秩1算法
        :return:
        """
        iter_, sol_tol, x_n = 0, np.infty, np.copy(self.x0)  # 初始化
        jacobi_mat = self.jacobi_obj.solve_jacobi_mat()  # 求解雅可比矩阵
        Ak = self.jacobi_obj.cal_jacobi_mat_values(jacobi_mat, self.x0)  # 雅可比矩阵值
        if self.method.lower() == "broyden":
            self._solve_broyden_(Ak, sol_tol, iter_, x_n)
        elif self.method.lower() == "broyden2th":
            self._solve_broyden_2th_(Ak, sol_tol, iter_, x_n)
        elif self.method.lower() == "invbroyden":
            self._solve_inv_broyden_(Ak, sol_tol, iter_, x_n)
        elif self.method.lower() == "invbroyden2th":
            self._solve_inv_broyden_2th_(Ak, sol_tol, iter_, x_n)
        else:
            raise ValueError("仅支持broyden、broyden2th、invbroyden和invbroyden2th算法.")

        self.roots = self.iter_roots_precision[-1][1]  # 满足精度的根
        # 最终解向量针对每个方程的精度, 调用雅可比矩阵类方法计算
        self.fxs_precision = \
            self.jacobi_obj.cal_fx_values(self.roots.reshape(-1, 1)).flatten()
        if self.is_plt:  # 是否可视化图像
            plt.figure(figsize=(14, 5))
            plt.subplot(121)
            self.plt_precision_convergence_curve("QuasiNewton \ " + self.method)
            plt.subplot(122)
            self.plt_roots_convergence_curve("QuasiNewton \ " + self.method)
            plt.show()
        return self.roots, self.fxs_precision

    def _solve_broyden_(self, Ak, sol_tol, iter_, x_n):
        """
        核心算法：Broyden秩1算法
        """
        while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
            x_b = np.copy(x_n)  # 更新解向量
            sol_xb = self.jacobi_obj.cal_fx_values(x_b)  # F(xk)，构成一个向量
            x_n = x_b - np.dot(np.linalg.inv(Ak), sol_xb)  # 迭代公式，下一次迭代值
            sol_xn = self.jacobi_obj.cal_fx_values(x_n)  # F(x_{k + 1})
            y_k, z_k = x_n - x_b, sol_xn - sol_xb  # 拟牛顿公式各参数计算
            Ak_term = np.linalg.norm(y_k)  # Broyden秩1公式的分母，范数，标量值
            if np.abs(Ak_term) < 1e-50 or np.linalg.norm(sol_xn) <= self.eps:  # 终止条件之一
                break  # 避免除零，以及当前F(xk)的2范数满足精度要求
            Ak = Ak + (z_k - np.dot(Ak, y_k)) * y_k.T / Ak_term  # Broyden秩1公式，修正
            sol_tol = np.linalg.norm(x_n - x_b)  # 更新迭代次数和精度
            iter_ += 1  # 迭代次数增一
            self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])

    def _solve_broyden_2th_(self, Ak, sol_tol, iter_, x_n):
        """
        核心算法：Broyden秩1第二方法
        """
        while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
            x_b = np.copy(x_n)  # 更新解向量
            sol_xb = self.jacobi_obj.cal_fx_values(x_b)  # F(xk)，构成一个向量
            x_n = x_b - np.dot(np.linalg.inv(Ak), sol_xb)  # 迭代公式
            sol_xn = self.jacobi_obj.cal_fx_values(x_n)  # F(x_{k + 1})
            y_k = x_n - x_b  # 拟牛顿公式各参数计算
            Ak_term = np.dot(sol_xn.T, y_k)  # 公式的分母
            if np.abs(Ak_term) < 1e-50 or np.linalg.norm(sol_xn) <= self.eps:  # 终止条件之一
                break  # 避免除零，以及当前F(xk)的2范数满足精度要求
            Ak = Ak + np.dot(sol_xn, sol_xn.T) / Ak_term  # Broyden秩1第二公式, 修正
            iter_, sol_tol = iter_ + 1, np.linalg.norm(x_n - x_b)  # 更新迭代次数和精度
            self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])

    def _solve_inv_broyden_(self, Ak, sol_tol, iter_, x_n):
        """
        核心算法：逆Broyden秩1算法
        """
        Hk = np.linalg.inv(Ak)  # Ak的逆矩阵
        while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
            x_b = np.copy(x_n)  # 更新解向量
            sol_xb = self.jacobi_obj.cal_fx_values(x_b)  # 解向量x_k的方程组的值，向量
            s_k = -np.dot(Hk, sol_xb)  # 逆Broyden算法, Hk * F(x_k)
            x_n = x_b + s_k  # 下一次迭代的解向量x_{k+1}
            sol_xn = self.jacobi_obj.cal_fx_values(x_n)  # F(x_{k + 1})
            z_k = sol_xn - sol_xb  # z_k = y_{k + 1} - y_k
            Hk_term = np.dot(np.dot(s_k.T, Hk), z_k)  # 公式分母，标量值
            if np.abs(Hk_term) < 1e-50 or np.linalg.norm(sol_xn) <= self.eps:
                break  # 避免除零，以及当前F(xk)的2范数满足精度要求
            Hk = Hk + np.dot(np.dot((s_k - np.dot(Hk, z_k)), s_k.T), Hk) / Hk_term  # 修正
            iter_, sol_tol = iter_ + 1, np.linalg.norm(x_n - x_b)  # 更新迭代次数和精度
            self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])

    def _solve_inv_broyden_2th_(self, Ak, sol_tol, iter_, x_n):
        """
        核心算法：逆broyden第二方法
        """
        Hk = np.linalg.inv(Ak)  # Ak的逆矩阵
        while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
            x_b = np.copy(x_n)  # 更新解向量
            sol_xb = self.jacobi_obj.cal_fx_values(x_b)  # F(x_k)
            x_n = x_b - np.dot(Hk, sol_xb)  # 下一次迭代的解向量x_{k+1}, 迭代公式
            sol_xn = self.jacobi_obj.cal_fx_values(x_n)  # F(x_{k + 1})
            y_k, z_k = x_n - x_b, sol_xn - sol_xb  # 各参数变量计算
            Hk_term = np.dot((y_k - np.dot(Hk, z_k)).T, z_k)  # 分母，标量值
            if abs(Hk_term) < 1e-50 or np.linalg.norm(sol_xn) <= self.eps:
                break  # 避免除零，以及当前F(xk)的2范数满足精度要求
            Hk = Hk + (y_k - np.dot(Hk, z_k)) * (y_k - np.dot(Hk, z_k)).T / Hk_term  # 修正
            iter_, sol_tol = iter_ + 1, np.linalg.norm(x_n - x_b)  # 更新迭代次数和精度
            self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])
