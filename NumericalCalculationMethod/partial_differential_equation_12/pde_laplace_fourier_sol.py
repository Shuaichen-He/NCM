# -*- coding: UTF-8 -*-
"""
@file_name: pde_laplace_fourier_sol.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from numerical_integration_04.gauss_legendre_int import GaussLegendreIntegration
from util_font import *


class PDELaplaceFourierSolution:
    """
    拉普拉斯方程的傅里叶解
    """

    def __init__(self, u_x0, u_xb, u_y0, u_ya, a, b, m, n, pde_model=None):
        self.u_x0, self.u_xb = u_x0, u_xb  # 初值条件u(x,0)、u(x,b)
        self.u_y0, self.u_ya = u_y0, u_ya  # 初值条件u(0,y)、u(a,y)
        self.a, self.b = a, b  # 求解空间的区间，起始区间端点为0
        self.m, self.n = m, n  # 傅里叶级数的项数
        self.pde_model = pde_model  # 解析解，不存在则不传
        self.u_xy = None  # 近似解表达式

    def solve_pde(self):
        """
        求解拉普拉斯方程的傅里叶解，采用高斯—勒让德一重积分计算
        :return:
        """
        A_n, B_n = np.zeros(self.m), np.zeros(self.m)  # 存储积分，表达式系数
        C_n, D_n = np.zeros(self.n), np.zeros(self.n)  # 存储积分，表达式系数
        # 如下根据边界系数，采用高斯—勒让德积分计算
        for i in range(1, self.m + 1):
            # 1.上边界
            c_ = 2 / (self.a * np.sinh(i * np.pi * self.b / self.a))
            int_fun_ = lambda x: np.sin(i * np.pi / self.a * x)
            int_fun_expr = lambda x: self.u_xb(x) * int_fun_(x)
            gli = GaussLegendreIntegration(int_fun_expr, [0, self.a], zeros_num=15)
            B_n[i - 1] = c_ * gli.fit_int()
            # 2.下边界
            int_fun_expr = lambda x: self.u_x0(x) * int_fun_(x)
            gli = GaussLegendreIntegration(int_fun_expr, [0, self.a], zeros_num=15)
            A_n[i - 1] = c_ * gli.fit_int()
        for i in range(1, self.n + 1):
            # 3.左边界
            c_ = 2 / (self.b * np.sinh(i * np.pi * self.a / self.b))
            int_fun_ = lambda y: np.sin(i * np.pi / self.b * y)
            int_fun_expr = lambda y: self.u_y0(y) * int_fun_(y)
            gli = GaussLegendreIntegration(int_fun_expr, [0, self.b], zeros_num=15)
            C_n[i - 1] = c_ * gli.fit_int()
            # 4.右边界
            int_fun_expr = lambda y: self.u_ya(y) * int_fun_(y)
            gli = GaussLegendreIntegration(int_fun_expr, [0, self.b], zeros_num=15)
            D_n[i - 1] = c_ * gli.fit_int()

        # 如下构造近似解表达式，采用符号形式
        x, y = sympy.symbols("x, y")
        u_xy = 0.0
        for i in range(1, self.m + 1):
            c_1 = i * np.pi / self.a
            term = A_n[i - 1] * sympy.sinh(c_1 * (self.b - y)) + B_n[i - 1] * sympy.sinh(c_1 * y)
            u_xy += sympy.sin(c_1 * x) * term
        for i in range(1, self.n + 1):
            c_2 = i * np.pi / self.b
            term = C_n[i - 1] * sympy.sinh(c_2 * (self.a - x)) + D_n[i - 1] * sympy.sinh(c_2 * x)
            u_xy += sympy.sin(c_2 * y) * term
        self.u_xy = sympy.lambdify((x, y), u_xy, "numpy")  # 转化为lambda函数
        return self.u_xy

    def plt_pde_laplace_surface(self, x_span, y_span):
        """
        可视化数值解，若存在解析解，则可视化误差曲面
        :return:
        """
        xi, yi = np.linspace(x_span[0], x_span[1], 50), np.linspace(y_span[0], y_span[1], 50)
        x, y = np.meshgrid(xi, yi)
        num_sol = self.u_xy(x, y)
        if self.pde_model:
            fig = plt.figure(figsize=(14, 5))
            ax = fig.add_subplot(121, projection='3d')
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca(projection='3d')
        ax.plot_surface(x, y, num_sol, cmap='rainbow')
        ax.set_xlabel("$x$", fontdict={"fontsize": 18})
        ax.set_ylabel("$y$", fontdict={"fontsize": 18})
        ax.set_zlabel("$U$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.title("$Laplace$方程$(Fourier)$解曲面", fontdict={"fontsize": 18})
        if self.pde_model:
            ax = fig.add_subplot(122, projection='3d')
            analytical_sol = self.pde_model(x, y)
            error_ = analytical_sol - num_sol  # 误差
            rmse = np.sqrt(np.mean(error_ ** 2))
            print("均方误差：", rmse)
            ax.plot_surface(x, y, error_, cmap='rainbow')
            z_format = plt.FormatStrFormatter('%.e')  # 设置y轴标签文本的格式
            ax.zaxis.set_major_formatter(z_format)
            ax.set_xlabel("$x$", fontdict={"fontsize": 18})
            ax.set_ylabel("$y$", fontdict={"fontsize": 18})
            # ax.set_zlabel("$Error$", fontdict={"fontsize": 18})
            plt.tick_params(labelsize=16)  # 刻度字体大小16
            plt.title("数值解误差曲面$(MSE=%.5e)$" % rmse, fontdict={"fontsize": 18})
        plt.show()
