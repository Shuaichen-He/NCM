# -*- coding: UTF-8 -*-
"""
@file_name: ode_ritz_galerkin.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
# 采用高斯—勒让德求积公式求解一重数值积分
from numerical_integration_04.gauss_legendre_int import GaussLegendreIntegration
# 采用平方根法求解线性方程组的解
from direct_solution_linear_equations_06.square_root_decomposition import SquareRootDecompositionAlgorithm
from util_font import *


class ODERitzGalerkin:
    """
    采用Ritz-Galerkin变元法求解二阶常微分方程，非稀疏
    """

    def __init__(self, f_ux, x_span, n, basis_func="poly", ode_model=None):
        self.f_ux = f_ux  # 右端函数，符号函数定义，固定自变量符号为x
        self.x_span = x_span  # 求解区间，支持[a, b]形式
        self.n = n  # 子空间数
        self.basis_func = basis_func  # 基函数类型
        self.ode_model = ode_model  # 解析解，不存在可不传
        self.ux = None  # 最终近似解表达式

    def fit_ode(self):
        """
        Ritz-Galerkin变元法
        :return:
        """
        a, b = self.x_span[0], self.x_span[1]
        x, k = sympy.symbols("x, k")  # 定义基函数的符号变量
        basis_func = None  # 定义基函数
        if self.basis_func.lower() == "poly":
            basis_func = (x - a) * (b - x) * (x ** (k - 1))  # 多项式基函数
        elif self.basis_func.lower() == "sin":
            basis_func = sympy.sin(k * np.pi * (x - a) / (b - a))  # 三角基函数
        else:
            print("基函数类型仅支持poly和sin.")
            exit(0)
        diff_bf = sympy.simplify(basis_func.diff(x, 1))  # 基函数的一阶导数

        # 如下构造求解近似解表达式的系数c的系数矩阵和右端向量
        c_mat = np.zeros((self.n, self.n))  # 系数矩阵
        b_vector = np.zeros(self.n)  # 右端向量
        for i in range(1, self.n + 1):
            phi_i, diff_phi_i = basis_func.subs({k: i}), diff_bf.subs({k: i})
            b_int_fun = sympy.lambdify(x, self.f_ux * phi_i)
            gli = GaussLegendreIntegration(b_int_fun, self.x_span, 15)  # 高斯—勒让德求积法
            b_vector[i - 1] = gli.fit_int()
            for j in range(i, self.n + 1):
                phi_j, diff_phi_j = basis_func.subs({k: j}), diff_bf.subs({k: j})
                int_fun = sympy.lambdify(x, diff_phi_i * diff_phi_j + phi_i * phi_j)  # 被积函数
                gli = GaussLegendreIntegration(int_fun, self.x_span, 15)  # 高斯—勒让德求积法
                c_mat[i - 1, j - 1] = gli.fit_int()
                c_mat[j - 1, i - 1] = c_mat[i - 1, j - 1]  # 对称矩阵

        # 采用平方根法求解线性方程组的解，正定对称矩阵
        srd = SquareRootDecompositionAlgorithm(c_mat, b_vector, sol_method="cholesky")
        c_sol = srd.fit_solve()  # 获得解
        # c_sol = np.linalg.solve(c_mat, b_vector)
        # 如下构造近似解表达式
        self.ux = 0.0
        for i in range(1, self.n + 1):
            self.ux += c_sol[i - 1] * basis_func.subs({k: i})
        self.ux = sympy.simplify(self.ux)
        return self.ux

    def plt_ode_curve(self):
        """
        可视化ODE数值解
        :return:
        """
        def plt_num_sol(xi, num_sol):
            plt.plot(xi, num_sol, "--", lw=2, label="$Numerical\ solution \ \hat y(x)$")
            plt.xlabel("$x$", fontdict={"fontsize": 18})
            plt.ylabel("$y(x) \ / \ \hat y(x)$", fontdict={"fontsize": 18})
            plt.title("$Ritz-Galerkin$变元法求解二阶$ODE$数值解", fontdict={"fontsize": 18})
            plt.legend(frameon=False, fontsize=16)
            plt.grid(ls=":")
            plt.tick_params(labelsize=18)  # 刻度字体大小16

        x = sympy.symbols("x")
        xi = np.linspace(self.x_span[0], self.x_span[1], 100)
        ux = sympy.lambdify(x, self.ux)
        num_sol = ux(xi)  # 数值解

        if self.ode_model is None:
            plt.figure(figsize=(7, 5))
            plt_num_sol(xi, num_sol)
            plt.show()
        else:
            analytical_sol = self.ode_model(xi)  # 解析解
            plt.figure(figsize=(14, 5))
            plt.subplot(121)
            plt.plot(xi, analytical_sol, "-", lw=2, label="$Analytical \ solution \ y(x)$")
            plt_num_sol(xi, num_sol)
            plt.subplot(122)
            error = analytical_sol - num_sol  # 误差
            error_norm = np.linalg.norm(error)  # 误差2范数
            plt.plot(xi, error, "-", lw=1.5)
            plt.xlabel("$x$", fontdict={"fontsize": 18})
            plt.ylabel("$\epsilon = y_k - \hat y_k$", fontdict={"fontsize": 18})
            plt.title("数值解的误差曲线$\Vert y - \hat{y} \Vert_2=%.5e$" % error_norm, fontsize=18)
            plt.grid(ls=":")
            plt.tick_params(labelsize=18)  # 刻度字体大小16
            plt.show()
