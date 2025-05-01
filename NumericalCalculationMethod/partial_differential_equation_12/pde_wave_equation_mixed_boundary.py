# -*- coding: UTF-8 -*-
"""
@file_name: pde_wave_equation_mixed_boundary.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class PDEWaveEquationMixedBoundary:
    """
    双曲型偏微分方程，波动方程求解
    """

    def __init__(self, fun_xt, alpha_fun, beta_fun, u_x0, du_x0,  # 波动方程的函数项
                 x_a, t_T, c, x_h, t_h, pde_model=None):
        self.fun_xt = fun_xt  # 波动方程的右端项函数
        # 初始边界条件函数, 对应u(0, y)和u(a, y)，关于y的函数
        self.alpha_fun, self.beta_fun = alpha_fun, beta_fun
        # 初始边界条件函数，对应u(x, 0)和u'(x, 0)，关于x的函数
        self.u_x0, self.du_x0 = u_x0, du_x0
        self.x_a, self.t_T = x_a, t_T  # 分别表示自变量x和t的求解区域右端点
        self.c = c  # 一维非齐次波动方程的常数项
        self.x_h, self.t_h = x_h, t_h  # 分别表示自变量x和t的求解步长
        self.n, self.m = int(self.x_a / self.x_h) + 1, int(self.t_T / self.t_h) + 1  # 划分网格区间点数
        self.u_xt = None  # 存储pde数值解
        self.pde_model = pde_model  # 解析解存在的情况下，可进行误差分析

    def solve_pde(self):
        """
        差分格式求解一维二阶非齐次波动方程的数值解
        :return:
        """
        r = self.c * self.t_h / self.x_h  # 差分格式系数常量
        if r > 1:
            raise ValueError("r = %.5f，非稳定格式，重新划分步长." % r)
        print("r = %.5f，稳定格式求解波动方程的数值解." % r)
        s1, s2 = 1 - r ** 2, r ** 2  # 差分格式的系数
        self.u_xt = np.zeros((self.n, self.m))  # 波动方程的数值解
        # 边界条件的处理
        xi, ti = np.linspace(0, self.x_a, self.n), np.linspace(0, self.t_T, self.m)
        self.u_xt[[0, -1], :] = self.alpha_fun(ti), self.beta_fun(ti)
        self.u_xt[:, 0] = self.u_x0(xi)
        # 根据边界情况，计算第2列
        self.u_xt[1:-1, 1] = s1 * self.u_x0(xi[1:-1]) + \
                             self.t_h * self.du_x0(xi[1:-1]) + \
                             s2 / 2 * (self.u_x0(xi[2:]) + self.u_x0(xi[:-2]))
        # 第3列递推到最后
        for j in range(2, self.m):
            self.u_xt[1:-1, j] = s2 * self.u_xt[:-2, j - 1] + \
                                 2 * s1 * self.u_xt[1:-1, j - 1] + \
                                 s2 * self.u_xt[2:, j - 1] - \
                                 self.u_xt[1:-1, j - 2] + \
                                 self.t_h ** 2 * self.fun_xt(ti[j - 1], xi[1:-1])
        return self.u_xt.T

    def plt_pde_wave_surface(self):  # 参考齐次波动方程
        """
        可视化数值解
        :return:
        """
        xi, ti = np.linspace(0, self.x_a, self.n), np.linspace(0, self.t_T, self.m)
        x, t = np.meshgrid(xi, ti)
        if self.pde_model:
            fig = plt.figure(figsize=(14, 5))
            ax = fig.add_subplot(121, projection='3d')
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca(projection='3d')
        ax.plot_surface(x, t, self.u_xt.T, cmap='rainbow')
        ax.set_xlabel("$x$", fontdict={"fontsize": 18})
        ax.set_ylabel("$t$", fontdict={"fontsize": 18})
        ax.set_zlabel("$U$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.title("一维二阶非齐次波动方程数值解曲面", fontdict={"fontsize": 18})
        if self.pde_model:
            ax = fig.add_subplot(122, projection='3d')
            analytical_sol = self.pde_model(x, t)
            error_ = analytical_sol - self.u_xt.T  # 误差
            ax.plot_surface(x, t, error_, cmap='rainbow')
            mae = np.mean(np.abs(error_))  # 均方根误差
            print("最大绝对值误差：%.10e" % np.max(np.abs(error_)))
            ax.set_xlabel("$x$", fontdict={"fontsize": 18})
            ax.set_ylabel("$t$", fontdict={"fontsize": 18})
            ax.set_zlabel("$\epsilon$", fontdict={"fontsize": 18})
            plt.tick_params(labelsize=16)  # 刻度字体大小16
            plt.title("误差曲面$\epsilon=U(x,t) - \hat U(x,t),\ MAE=%.3e$" % mae, fontdict={"fontsize": 18})
        plt.show()

    def plt_pde_wave_curve_contourf(self):  # 参考齐次波动方程
        """
        可视化某些时刻的数值解，以及等值线图
        :return:
        """
        # 1、不同时刻的波的传播随空间坐标的变化
        xi, ti = np.linspace(0, self.x_a, self.n), np.linspace(0, self.t_T, self.m)
        idx = np.array([1, len(ti) / 4, len(ti) / 2, 3 * len(ti) / 4, len(ti)], np.int) - 1
        plt.figure(figsize=(14, 5))
        plt.subplot(121)
        for i in idx:
            plt.plot(xi, self.u_xt[:, i], label='$t=%.5f$' % ti[i], lw=1.5)
        plt.ylabel('$U(x,t)$', fontdict={"fontsize": 18})
        plt.xlabel('$x$', fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16, loc='best')
        plt.title("波动方程在某些时刻的数值解", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        # 2、带有填充区域的等值线图
        plt.subplot(122)
        extent = [0, self.t_T + self.t_h, 0, self.x_a + self.x_h]  # 时间和空间的取值范围
        plt.contourf(self.u_xt, levels=15, origin='lower', extent=extent, cmap=plt.get_cmap("jet"))
        plt.colorbar()  # 颜色bar
        plt.ylabel('$x$', fontdict={"fontsize": 18})
        plt.xlabel('$t$', fontdict={"fontsize": 18})
        plt.title("波动方程在时刻$t = %.1f$的等值线图" % self.t_T, fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.show()
