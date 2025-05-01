# -*- coding: UTF-8 -*-
"""
@file_name: pde_poisson_equation_sor.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class PDEPoissonEquationSORIteration:
    """
    迭代法求解泊松方程, Dirichlet边界条件
    """

    def __init__(self, g_xy, f_ux0, f_uxb, f_u0y, f_uay,  # 函数项
                 x_a, y_b, x_h, y_h, eps=1e-3, max_iter=200, pde_model=None):
        self.g_xy = g_xy  # 泊松方程右端函数
        self.f_ux0, self.f_uxb = f_ux0, f_uxb  # 初始边界条件函数
        self.f_u0y, self.f_uay = f_u0y, f_uay  # 初始边界条件函数
        self.x_a, self.y_b = x_a, y_b  # 分别表示自变量x和y的求解区域右端点
        self.x_h, self.y_h = x_h, y_h  # 分别表示自变量x和y的求解步长
        self.n, self.m = int(self.x_a / self.x_h) + 1, int(self.y_b / self.y_h) + 1  # 划分网格区间点数
        self.u_xy = None  # 存储pde数值解
        self.pde_model = pde_model  # 解析解存在的情况下，可进行误差分析
        self.eps, self.max_iter = eps, max_iter  # 迭代法求解精度和最大迭代次数

    def solve_pde(self):
        """
        迭代法求解泊松方程
        :return:
        """
        ave = (self.x_a * (self.f_ux0(0) + self.f_uxb(0)) +
               self.y_b * (self.f_u0y(0) + self.f_uay(0))) / (2 * (self.x_a + self.y_b))
        self.u_xy = ave * np.ones((self.n, self.m))  # 初始数值解
        # 边界条件
        xi, yi = np.linspace(0, self.x_a, self.n), np.linspace(0, self.y_b, self.m)  # 等分x和y
        self.u_xy[0, :] = self.f_u0y(yi)  # 左边界
        self.u_xy[-1, :] = self.f_uay(yi)  # 右边界
        self.u_xy[:, 0] = self.f_ux0(xi)  # 底部
        self.u_xy[:, -1] = self.f_uxb(xi)  # 顶部

        # 松弛因子计算
        w = 4 / (2 + np.sqrt(4 - (np.cos(np.pi / (self.n - 1)) +
                                  np.cos(np.pi / (self.m - 1))) ** 2))

        # 松弛迭代法求解
        err, iter_ = 1, 0  # 初始化误差和迭代次数
        while err > self.eps and iter_ < self.max_iter:
            err, iter_ = 0.0, iter_ + 1
            for j in range(1, self.m - 1):
                for i in range(1, self.n - 1):
                    ut = self.u_xy[i, j + 1] + self.u_xy[i, j - 1] + \
                         self.u_xy[i + 1, j] + \
                         self.u_xy[i - 1, j] - 4 * self.u_xy[i, j]  # 子项
                    relax = w * (ut + self.x_h * self.y_h * self.g_xy(xi[i], yi[j])) / 4
                    self.u_xy[i, j] += relax
                    if err <= abs(relax):
                        err = abs(relax)
        print("iter_num = %d，max r_ij = %.10e" % (iter_, err))
        return self.u_xy.T

    def plt_pde_poisson_surface(self):  # 参考一维波动方程
        """
        可视化数值解
        :return:
        """
        xi, yi = np.linspace(0, self.x_a, self.n), np.linspace(0, self.y_b, self.m)
        x, y = np.meshgrid(xi, yi)
        if self.pde_model:
            fig = plt.figure(figsize=(14, 5))
            ax = fig.add_subplot(121, projection='3d')
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca(projection='3d')
        ax.plot_surface(x, y, self.u_xy.T, cmap='rainbow')
        ax.set_xlabel("$x$", fontdict={"fontsize": 18})
        ax.set_ylabel("$y$", fontdict={"fontsize": 18})
        ax.set_zlabel("$U$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.title("$Poisson$方程($Dirichlet$)数值解曲面", fontdict={"fontsize": 18})
        if self.pde_model:
            ax = fig.add_subplot(122, projection='3d')
            analytical_sol = self.pde_model(x, y)
            error_ = analytical_sol - self.u_xy.T  # 误差
            ax.plot_surface(x, y, error_, cmap='rainbow')
            mae = np.mean(np.abs(error_))  # 平均绝对值误差
            print("平均绝对值误差：%.10e" % mae)
            print("最大绝对值误差：%.10e" % np.max(np.abs(error_)))
            ax.set_xlabel("$x$", fontdict={"fontsize": 18})
            ax.set_ylabel("$y$", fontdict={"fontsize": 18})
            # ax.set_zlabel("$\epsilon$", fontdict={"fontsize": 18})
            # z_format = plt.FormatStrFormatter('%.e')  # 设置y轴标签文本的格式
            # ax.zaxis.set_major_formatter(z_format)
            plt.tick_params(labelsize=16)  # 刻度字体大小16
            plt.title("$\epsilon=U(x,y) - \hat U(x,y),\ MAE=%.3e$" % mae, fontdict={"fontsize": 18})
        plt.show()
