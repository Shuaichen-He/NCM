# -*- coding: UTF-8 -*-
"""
@file:rank2_quasi_newton_jm.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
import matplotlib.pyplot as plt
from nonlinear_equations_09.utils.nonlinear_equations_utils import NonLinearEquationsUtils
from nonlinear_equations_09.jacobi_matrix import JacobiMatrix


class Rank2QuasiNewton(NonLinearEquationsUtils):
    """
    对称秩算法求解非线性方程组的解，包括对称秩2算法中的DFP、BFS、BFGS和逆BFGS四种算法,
    继承NonLinearEquationsUtils, 采用雅可比矩阵的逆矩阵作为H0
    """

    def __init__(self, nlin_fxs, sym_vars, x0, max_iter=200, eps=1e-10,
                 method="dfp", is_plt=False):
        nlin_equs_expr = sympy.lambdify([sym_vars], nlin_fxs, "numpy")  # 转换为数值方程
        NonLinearEquationsUtils.__init__(self, nlin_equs_expr, x0, max_iter, eps, is_plt)
        self.n = len(x0)  # 解的个数
        self.method = method  # 秩2四种算法
        self.jacobi_obj = JacobiMatrix(nlin_fxs, sym_vars)  # 初始化雅可比矩阵对象
        self.fxs_precision = None  # 最终解向量针对每个方程的精度

    def fit_nlin_roots(self):
        """
        核心算法: 秩2迭代法求解非线性方程组的解，根据方法选择对应的秩2算法
        :return:
        """
        iter_, sol_tol, x_n = 0, np.infty, np.copy(self.x0)  # 初始化
        jacobi_mat = self.jacobi_obj.solve_jacobi_mat()  # 求解雅可比矩阵
        Ak = self.jacobi_obj.cal_jacobi_mat_values(jacobi_mat, self.x0)  # 雅可比矩阵值
        Hk = np.linalg.inv(Ak)  # 初始化为雅可比矩阵的逆
        if self.method.lower() == "dfp":
            self._solve_DFP_rank2_(Hk, sol_tol, iter_, x_n)
        elif self.method.lower() == "bfs":
            self._solve_BFS_rank2_(Hk, sol_tol, iter_, x_n)
        elif self.method.lower() == "bfgs":
            self._solve_BFGS_rank2_(Ak, sol_tol, iter_, x_n)
        elif self.method.lower() == "invbfgs":
            self._solve_inv_BFGS_rank2_(Hk, sol_tol, iter_, x_n)
        else:
            raise ValueError("仅支持DFP、BFS、BFGS和逆BFGS算法.")
        self.roots = self.iter_roots_precision[-1][1]  # 满足精度的根
        # 最终解向量针对每个方程的精度
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

    def _solve_DFP_rank2_(self, Hk, sol_tol, iter_, x_n):
        """
        核心算法：秩2 DFP算法
        """
        while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
            x_b = np.copy(x_n)  # 更新数值解，向量形式，深拷贝
            sol_xb = self.jacobi_obj.cal_fx_values(x_b)  # 上一次数值解的函数值向量F(x_k)
            x_n = x_b - np.dot(Hk, sol_xb)  # 计算新的数值解向量x_{k+1}
            sol_xn = self.jacobi_obj.cal_fx_values(x_n)  # 新的数值解的函数值向量F(x_{k+1})
            if np.linalg.norm(sol_xn) <= self.eps:
                break  # 终止条件||F(x_(k+1))|| < eps
            s_k, y_k = x_n - x_b, sol_xn - sol_xb  # 求解Bk修正计算中间变量
            # if np.dot(s_k.T, y_k) > 0:  # 对如下计算，放宽条件
            Hk_term1, Hk_term2 = np.dot(y_k.T, s_k), np.dot(np.dot(y_k.T, Hk), y_k)  # 修正公式中的分母
            if np.abs(Hk_term1) <= 1e-50 or np.abs(Hk_term2) <= 1e-50:
                break  # 避免被零除
            Hk = Hk + np.dot(s_k, s_k.T) / Hk_term1 - \
                 np.dot(np.dot(np.dot(Hk, y_k), y_k.T), Hk) / Hk_term2  # Hk的修正
            iter_, sol_tol = iter_ + 1, np.linalg.norm(x_n - x_b)  # 相邻解的2范数作为终止条件
            self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])  # 存储迭代过程数值

    def _solve_BFS_rank2_(self, Hk, sol_tol, iter_, x_n):
        """
        核心算法：秩2 BFS算法
        """
        while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
            x_b = np.copy(x_n)  # 解向量的更新
            sol_xb = self.jacobi_obj.cal_fx_values(x_b)  # F(xk)
            x_n = x_b - np.dot(Hk, sol_xb)  # 迭代公式
            sol_xn = self.jacobi_obj.cal_fx_values(x_n)  # F(x_(k+1))
            s_k, y_k = x_n - x_b, sol_xn - sol_xb  # 修正矩阵的各参数计算
            Hk_term = np.dot(s_k.T, y_k)  # BFS修正公式中的分母
            if abs(Hk_term) < 1e-50 or np.linalg.norm(sol_xn) <= self.eps:
                break  # 避免被零除, 以及终止条件||F(x_(k+1))|| < eps
            # if np.dot(s_k.T, y_k) > 0:  # 对如下计算，放宽条件
            uk = 1 + np.dot(np.dot(y_k.T, Hk), y_k) / np.dot(s_k.T, y_k)  # 标量
            Hk = Hk + (np.dot(uk * s_k, s_k.T) - np.dot(np.dot(Hk, y_k), s_k.T)
                       - np.dot(np.dot(s_k, y_k.T), Hk)) / Hk_term  # 修正
            iter_, sol_tol = iter_ + 1, np.linalg.norm(x_n - x_b)  # 更新迭代次数和精度
            self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])  # 存储

    def _solve_BFGS_rank2_(self, Hk, sol_tol, iter_, x_n):
        """
        核心算法：秩2 BFGS算法
        """
        while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
            x_b = np.copy(x_n)  # 解向量的更新
            sol_xb = self.jacobi_obj.cal_fx_values(x_b)  # F(xk)
            x_n = x_b - np.dot(np.linalg.inv(Hk), sol_xb)  # 迭代公式
            sol_xn = self.jacobi_obj.cal_fx_values(x_n)  # F(x_(k+1))
            if np.linalg.norm(sol_xn) <= self.eps:
                break  # 增加终止条件||F(x_(k+1))|| < eps
            s_k, y_k = x_n - x_b, sol_xn - sol_xb  # 修正矩阵的各参数计算
            Hk_term1, Hk_term2 = np.dot(y_k.T, s_k), np.dot(np.dot(s_k.T, Hk), s_k)  # 分母
            if np.abs(Hk_term1) < 1e-50 or np.abs(Hk_term2) < 1e-50:  # 避免被零除
                break
            Hk = Hk + np.dot(y_k, y_k.T) / Hk_term1 - \
                 np.dot(np.dot(np.dot(Hk, s_k), s_k.T), Hk) / Hk_term2  # 修正
            iter_, sol_tol = iter_ + 1, np.linalg.norm(x_n - x_b)  # 更新迭代次数和精度
            self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])  # 存储

    def _solve_inv_BFGS_rank2_(self, Hk, sol_tol, iter_, x_n):
        """
        核心算法：秩2 逆BFGS算法
        """
        while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
            x_b = np.copy(x_n)  # 解向量的更新
            sol_xb = self.jacobi_obj.cal_fx_values(x_b)  # F(xk)
            x_n = x_b - np.dot(Hk, sol_xb)  # 迭代公式
            sol_xn = self.jacobi_obj.cal_fx_values(x_n)  # F(x_(k+1))
            s_k, y_k = x_n - x_b, sol_xn - sol_xb  # 修正矩阵的各参数计算
            Hk_term = np.dot(s_k.T, y_k)  # 公式后两式的分母
            if np.abs(Hk_term) < 1e-50 or np.linalg.norm(sol_xn) <= self.eps:
                break  # 避免被零除, 以及增加终止条件||F(x_(k+1))|| < eps
            term1 = np.dot(s_k - np.dot(Hk, y_k), s_k.T) + \
                    np.dot(s_k, (s_k - np.dot(Hk, y_k)).T)
            term2 = np.dot((s_k - np.dot(Hk, y_k)).T, y_k) * np.dot(s_k, s_k.T)
            Hk = Hk + term1 / Hk_term - term2 / Hk_term ** 2  # 修正
            iter_, sol_tol = iter_ + 1, np.linalg.norm(x_n - x_b)  # 更新迭代次数和精度
            self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])  # 存储
