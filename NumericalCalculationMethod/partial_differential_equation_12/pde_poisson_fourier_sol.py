# -*- coding: UTF-8 -*-
"""
@file_name: pde_poisson_fourier_sol.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from numerical_integration_04.gauss_legendre_int import GaussLegendreIntegration
from numerical_integration_04.gauss_legendre_2d_integration import GaussLegendreDoubleIntegration
from util_font import *


class PDEPoissonFourierSolution:
    """
    泊松方程的傅里叶解
    """

    def __init__(self, f_xy, a, b, m, n, pde_model=None):
        self.f_xy = f_xy  # 方程右端方程f(x,y)
        self.a, self.b = a, b  # 求解空间的区间，起始区间端点为0
        self.m, self.n = m, n  # 傅里叶级数的项数
        self.pde_model = pde_model  # 解析解，不存在则不传
        self.u_xy = None  # 近似解表达式

    def solve_pde(self):
        """
        求解泊松方程的傅里叶解，采用高斯—勒让德一重、二重积分计算
        :return:
        """
        E_mn = np.zeros((self.m, self.n))
        for i in range(1, self.m + 1):
            for j in range(1, self.n + 1):
                lambda_ = (i * np.pi / self.a) ** 2 + (j * np.pi / self.b) ** 2
                ini_fun = lambda x, y: np.sin(i * np.pi / self.a * x) * \
                                       np.sin(j * np.pi / self.b * y)
                int_fun_expr = lambda x, y: self.f_xy(x, y) * ini_fun(x, y)
                g2dli = GaussLegendreDoubleIntegration(int_fun_expr, [0, self.a],
                                                       [0, self.b], zeros_num=15)
                E_mn[i - 1, j - 1] = -4 / (self.a * self.b * lambda_) * \
                                     g2dli.cal_2d_int()

        # 如下构造近似解表达式，采用符号形式
        x, y = sympy.symbols("x, y")
        u_xy = 0.0
        for i in range(1, self.m + 1):
            for j in range(1, self.n + 1):
                u_xy += E_mn[i - 1, j - 1] * sympy.sin(i * np.pi / self.a * x) * \
                        sympy.sin(j * np.pi / self.b * y)
        self.u_xy = sympy.lambdify((x, y), u_xy, "numpy")
        return self.u_xy

    def plt_pde_poisson_surface(self, x_span, y_span):  # 参考拉普拉斯方程傅里叶解代码
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
        plt.title("$Poisson$方程$(Fourier)$解曲面", fontdict={"fontsize": 18})
        if self.pde_model:
            ax = fig.add_subplot(122, projection='3d')
            analytical_sol = self.pde_model(x, y)
            error_ = analytical_sol - num_sol  # 误差
            ax.plot_surface(x, y, error_, cmap='rainbow')
            mae = np.mean(np.abs(error_))  # 平均绝对值误差
            print("平均绝对值误差：%.10e" % mae)
            print("最大绝对值误差：%.10e" % np.max(np.abs(error_)))
            ax.set_xlabel("$x$", fontdict={"fontsize": 18})
            ax.set_ylabel("$y$", fontdict={"fontsize": 18})
            ax.set_zlabel("$\epsilon$", fontdict={"fontsize": 18})
            # z_format = plt.FormatStrFormatter('%.e')  # 设置y轴标签文本的格式
            # ax.zaxis.set_major_formatter(z_format)
            plt.tick_params(labelsize=16)  # 刻度字体大小16
            plt.title("$\epsilon=U(x,y) - \hat U(x,y),\ MAE=%.3e$" % mae, fontdict={"fontsize": 18})
        plt.show()

