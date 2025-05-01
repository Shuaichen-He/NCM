# -*- coding: UTF-8 -*-
"""
@file_name: pde_2d_wave_fourier_sol.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from numerical_integration_04.gauss_legendre_2d_integration import GaussLegendreDoubleIntegration
from util_font import *


class PDE2DWaveFourierSolution:
    """
    二维波动方程的傅里叶解
    """

    def __init__(self, f_xyt_0, df_xyt_0, c, a, b, t_T, m, n, pde_model=None):
        self.f_xyt_0 = f_xyt_0  # 初值条件f(x,y,0)
        self.df_xyt_0 = df_xyt_0  # 初值条件，对应f(x,y,0)对t的一阶导在t=0时刻的函数
        self.c = c  # 二维波动方程的系数，c ** 2形式
        self.a, self.b = a, b  # 求解空间的区间，起始区间端点为0
        self.t_T = t_T  # 求解时刻
        self.m, self.n = m, n
        self.pde_model = pde_model  # 解析解，不存在则不传
        self.u_xyt = None  # 近似解表达式

    def solve_pde(self):
        """
        求解二维波动方程的傅里叶解，采用高斯—勒让德二重积分计算
        :return:
        """
        B_mn = np.zeros((self.m, self.n))  # 存储积分，表达式系数
        B_mn_d = np.zeros((self.m, self.n))  # 存储积分，表达式系数
        lambda_ = np.zeros((self.m, self.n))  # 存储指数项系数
        for i in range(self.m):
            for j in range(self.n):
                lambda_[i, j] = self.c * np.pi * \
                                np.sqrt((i / self.a) ** 2 + (j / self.b) ** 2)
                # 如下求解Bmn和B(*)mn系数，采用高斯—勒让德二重积分计算
                int_fun_ = lambda x, y: np.sin(i / self.a * np.pi * x) * \
                                        np.sin(j / self.b * np.pi * y)
                int_fun_expr = lambda x, y: self.f_xyt_0(x, y) * int_fun_(x, y)
                gl2di = GaussLegendreDoubleIntegration(int_fun_expr, [0, self.a],
                                                       [0, self.b], zeros_num=15)
                B_mn[i, j] = 4 / (self.a * self.b) * gl2di.cal_2d_int()
                if self.df_xyt_0(self.a, self.b):  # g(x,y,0)为0时不计算
                    int_gun_expr = lambda x, y: self.df_xyt_0(x, y) * int_fun_(x, y)
                    gl2di = GaussLegendreDoubleIntegration(int_gun_expr, [0, self.a],
                                                           [0, self.b], zeros_num=15)
                    B_mn_d[i, j] = 4 / (self.a * self.b * lambda_[i, j]) * \
                                   gl2di.cal_2d_int()
        # 如下构造近似解表达式，采用符号形式
        x, y, t = sympy.symbols("x, y, t")
        u_xyt = 0.0
        for i in range(self.m):
            for j in range(self.n):
                c_ = B_mn[i, j] * sympy.cos(lambda_[i, j] * t) + B_mn_d[i, j] * \
                     sympy.sin(lambda_[i, j] * t)
                u_xyt += c_ * sympy.sin(i / self.a * np.pi * x) * \
                         sympy.sin(j / self.b * np.pi * y)
        print(u_xyt)
        self.u_xyt = sympy.lambdify((x, y, t), u_xyt)
        return self.u_xyt

    def plt_pde_wave_surface(self):
        """
        可视化数值解，存在解析解，则可视化误差曲面
        :return:
        """
        xi, yi = np.linspace(0, self.a, 50), np.linspace(0, self.b, 50)
        x_, y_ = np.meshgrid(xi, yi)
        if self.pde_model:
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(121, projection='3d')
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca(projection='3d')
        num_sol = self.u_xyt(x_, y_, self.t_T)  # 数值解
        ax.plot_surface(x_, y_, num_sol, cmap='rainbow')
        ax.set_xlabel("$x$", fontdict={"fontsize": 18})
        ax.set_ylabel("$y$", fontdict={"fontsize": 18})
        ax.set_zlabel("$U$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.title("二维波动方程傅里叶解曲面", fontdict={"fontsize": 18})
        if self.pde_model:
            ax = fig.add_subplot(122, projection='3d')
            analytical_sol = self.pde_model(x_, y_, self.t_T)  # 解析解
            error_ = analytical_sol - num_sol  # 误差
            ax.plot_surface(x_, y_, error_, cmap='rainbow')
            mae = np.mean(np.abs(error_))  # 均方根误差
            print("最大绝对值误差：%.10e" % np.max(np.abs(error_)))
            ax.set_xlabel("$x$", fontdict={"fontsize": 18})
            ax.set_ylabel("$y$", fontdict={"fontsize": 18})
            ax.set_zlabel("$\epsilon$", fontdict={"fontsize": 18})
            plt.tick_params(labelsize=16)  # 刻度字体大小16
            plt.title("误差曲面$\epsilon=U(x,y,t) - \hat U(x,y,t),\ MAE=%.3e$" % mae, fontdict={"fontsize": 18})
        plt.show()
