# -*- coding: UTF-8 -*-
"""
@file_name: ode_ritz_galerkin_FEM.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
# 采用高斯—勒让德求积公式求解一重数值积分
from numerical_integration_04.gauss_legendre_int import GaussLegendreIntegration
from numerical_integration_04.composite_quadrature_formula import CompositeQuadratureFormula
# 采用追赶法求解线性方程组的解
from direct_solution_linear_equations_06.chasing_method_tridiagonal_matrix \
    import ChasingMethodTridiagonalMatrix
from util_font import *


class ODERitzGalerkinFEM:
    """
    采用Ritz-Galerkin有限元法求解二阶常微分方程，稀疏
    """

    def __init__(self, f_ux, x_span, n, ode_model=None):
        self.f_ux = f_ux  # 右端函数，符号函数定义，固定自变量符号为x
        self.x_span = x_span  # 求解区间，支持[a, b]形式
        self.n = n  # 子空间数
        self.ode_model = ode_model  # 解析解，不存在可不传
        self.ux = None  # 最终近似解表达式

    @staticmethod
    def basic_func(a, b, h):
        """
        构造有限元空间基函数
        :param a: [x_(i-1), x_(i)]的左端点x_(i-1)
        :param b: [x_(i-1), x_(i)]的右端点x_(i)
        :param h: 单元剖分步长
        :return:
        """
        x = sympy.symbols("x")
        # 构造分段基函数
        return sympy.Piecewise(((x - a) / h, (x >= a) & (x < b)),
                               ((b + h - x) / h, (x >= b) & (x <= b + h)),
                               (0.0, (x < a) | (x > b + h)), (0, True))

    def fit_ode(self):
        """
        Ritz-Galerkin有限元法
        :return:
        """
        a, b = self.x_span[0], self.x_span[1]
        xi = np.linspace(a, b, self.n + 2)  # 区间剖分点
        h = (b - a) / (self.n + 1)  # 单元剖分步长
        x, s = sympy.symbols("x, s")  # 定义符号变量，s为积分符号
        # 如下构造求解近似解表达式的系数c的系数矩阵和右端向量
        main_diag = np.zeros(self.n)  # 系数矩阵主对角线元素
        sub_diag = np.zeros(self.n - 1)  # 次对角线元素
        b_vector = np.zeros(self.n)  # 右端向量
        for i in range(self.n):
            b_int_fun_1 = self.f_ux.subs({x: (xi[i] + h * s)}) * s
            b_int_fun_2 = self.f_ux.subs({x: (xi[i + 1] + h * s)}) * (1 - s)
            # b_int_fun = sympy.lambdify(s, b_int_fun_1 + b_int_fun_2)
            b_int_fun = b_int_fun_1 + b_int_fun_2
            cqf = CompositeQuadratureFormula(b_int_fun, self.x_span, interval_num=30, int_type="cotes")
            cqf.fit_int()
            b_vector[i] = cqf.int_value
            # gli = GaussLegendreIntegration(b_int_fun, self.x_span, 15)  # 高斯—勒让德求积法
            # b_vector[i] = gli.fit_int()
            # 主对角线元素计算
            # int_fun = sympy.lambdify(s, 2 / h + h * (s ** 2 + (1 - s) ** 2))
            # gli = GaussLegendreIntegration(int_fun, self.x_span, 15)  # 高斯—勒让德求积法
            # main_diag[i] = gli.fit_int() / h
            int_fun = 2 / h + h * (s ** 2 + (1 - s) ** 2)
            cqf = CompositeQuadratureFormula(int_fun, self.x_span, interval_num=30, int_type="cotes")
            cqf.fit_int()
            main_diag[i] = cqf.int_value / h
        for i in range(self.n - 1):
            # 次对角线元素计算
            # int_fun = sympy.lambdify(s, -1 / h + h * (1 - s) * s)
            int_fun = -1 / h + h * (1 - s) * s
            # gli = GaussLegendreIntegration(int_fun, self.x_span, 15)  # 高斯—勒让德求积法
            # sub_diag[i] = gli.fit_int() / h
            cqf = CompositeQuadratureFormula(int_fun, self.x_span, interval_num=30, int_type="cotes")
            cqf.fit_int()
            sub_diag[i] = cqf.int_value / h
        # print(b_vector)
        # print(main_diag)
        # print(sub_diag)
        # 采用追赶法求解三对角线性方程组的解
        ctm = ChasingMethodTridiagonalMatrix(sub_diag, main_diag, sub_diag, b_vector)
        c_sol = ctm.fit_solve()  # 获得解
        # print(c_sol)
        # 如下构造近似解表达式
        self.ux = sympy.zeros(self.n)
        for i in range(self.n):
            print(self.basic_func(xi[i], xi[i + 1], h))
            self.ux[i] = c_sol[i] * self.basic_func(xi[i], xi[i + 1], h)
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
            plt.title("$Ritz-Galerkin$有限元法求解二阶$ODE$数值解", fontdict={"fontsize": 18})
            plt.legend(frameon=False, fontsize=16)
            plt.grid(ls=":")
            plt.tick_params(labelsize=18)  # 刻度字体大小16

        x = sympy.symbols("x")
        ux = []  # 针对每个基函数，转化为lambda函数，然后进行数值运算
        for i in range(self.n):
            ux.append(sympy.lambdify(x, self.ux[i], "numpy"))
        num_sol = np.zeros(100)  # 存储数值解
        xi = np.linspace(self.x_span[0], self.x_span[1], 100)
        for i in range(self.n):
            num_sol += ux[i](xi)
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
            error = analytical_sol - num_sol
            error_norm = np.linalg.norm(error)
            plt.plot(xi, error, "-", lw=1.5)
            plt.xlabel("$x$", fontdict={"fontsize": 18})
            plt.ylabel("$\epsilon = y_k - \hat y_k$", fontdict={"fontsize": 18})
            plt.title("数值解的误差曲线$\Vert y - \hat{y} \Vert_2=%.5e$" % error_norm, fontsize=18)
            plt.grid(ls=":")
            plt.tick_params(labelsize=18)  # 刻度字体大小16
            plt.show()
