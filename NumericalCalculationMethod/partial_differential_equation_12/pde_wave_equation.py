# -*- coding: UTF-8 -*-
"""
@file_name: pde_wave_equation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class PDEWaveEquation:
    """
    双曲型偏微分方程，波动方程求解
    """

    def __init__(self, f_fun, g_fun, b_u0t_fun, x_a, t_b, c, x_h, t_h, pde_model=None):
        self.f_fun, self.g_fun = f_fun, g_fun  # 初始边界条件函数
        self.b_u0t = b_u0t_fun  # u(0, t)和u(a, t)的函数
        self.x_a, self.t_b = x_a, t_b  # 分别表示自变量x和t的求解区域右端点，可扩展为区域[a, b]形式定义
        self.c = c  # 一维齐次波动方程的常数项
        self.x_h, self.t_h = x_h, t_h  # 分别表示自变量x和t的求解步长
        self.n, self.m = int(self.x_a / self.x_h) + 1, int(self.t_b / self.t_h) + 1  # 划分网格区间点数
        self.u_xt = None  # 存储pde数值解
        self.pde_model = pde_model  # 解析解存在的情况下，可进行误差分析

    def cal_pde(self):
        """
        差分格式求解一维齐次波动方程的数值解
        :return:
        """
        r = self.c * self.t_h / self.x_h  # 差分格式系数常量
        if r > 1:
            raise ValueError("r = %.5f，非稳定格式，重新划分步长." % r)
        print("r = %.5f，稳定格式求解波动方程的数值解." % r)
        cf_1, cf_2 = 2 - 2 * r ** 2, r ** 2  # 差分格式的系数
        self.u_xt = np.zeros((self.n, self.m))  # 波动方程的数值解
        ti = np.linspace(0, self.t_b, self.m)
        self.u_xt[0, :], self.u_xt[-1, :] = self.b_u0t[0](ti), self.b_u0t[1](ti)  # 边界条件
        # 初始条件，第一列和第二列数值，数值解按例存储
        i = np.arange(1, self.n - 1)
        self.u_xt[1:-1, 0] = self.f_fun(i * self.x_h)
        self.u_xt[1:-1, 1] = cf_1 / 2 * self.f_fun(i * self.x_h) + \
                             self.t_h * self.g_fun(i * self.x_h) + \
                             cf_2 / 2 * (self.f_fun((i + 1) * self.x_h) +
                                         self.f_fun((i - 1) * self.x_h))
        # 差分公式求解数值解，j = 1, 2, ...
        for j in range(1, self.m - 1):
            self.u_xt[1:-1, j + 1] = cf_1 * self.u_xt[1:-1, j] + \
                                     cf_2 * (self.u_xt[2:, j] + self.u_xt[:-2, j]) - \
                                     self.u_xt[1:-1, j - 1]

    def plt_pde_wave_surface(self):
        """
        可视化数值解
        """
        xi, ti = np.linspace(0, self.x_a, self.n), np.linspace(0, self.t_b, self.m)
        x, t = np.meshgrid(xi, ti)
        if self.pde_model:
            fig = plt.figure(figsize=(14, 5))
            ax = fig.add_subplot(121, projection='3d')
        else:
            fig = plt.figure(figsize=(7, 5))
            ax = plt.gca(projection='3d')
        ax.plot_surface(x, t, self.u_xt.T, cmap='rainbow')
        ax.set_xlabel("$x$", fontdict={"fontsize": 18})
        ax.set_ylabel("$t$", fontdict={"fontsize": 18})
        ax.set_zlabel("$U$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.title("一维二阶齐次波动方程数值解曲面", fontdict={"fontsize": 18})
        if self.pde_model:
            ax = fig.add_subplot(122, projection='3d')
            analytical_sol = self.pde_model(x, t)
            error_ = analytical_sol - self.u_xt.T  # 误差
            mae = np.mean(np.abs(error_))  # 均方根误差
            print("最大绝对值误差：%.10e" % np.max(np.abs(error_)))
            ax.plot_surface(x, t, error_, cmap='rainbow')
            # z_format = plt.FormatStrFormatter('%.e')  # 设置y轴标签文本的格式
            # ax.zaxis.set_major_formatter(z_format)
            ax.set_xlabel("$x$", fontdict={"fontsize": 18})
            ax.set_ylabel("$t$", fontdict={"fontsize": 18})
            ax.set_zlabel("$\epsilon$", fontdict={"fontsize": 18})
            plt.tick_params(labelsize=16)  # 刻度字体大小16
            plt.title("误差曲面$\epsilon=U(x,t) - \hat U(x,t),\ MAE=%.3e$" % mae, fontdict={"fontsize": 18})
        plt.show()

    def plt_pde_wave_curve_contourf(self):
        """
        可视化某些时刻的数值解，以及等值线图
        :return:
        """
        # 1、不同时刻的波的传播随空间坐标的变化
        xi = np.linspace(0, self.x_a, self.n)
        ti = np.linspace(0, self.t_b, self.m)
        idx = np.array([1, len(ti) / 4, len(ti) / 2, 3 * len(ti) / 4, len(ti)], np.int) - 1
        plt.figure(figsize=(14, 5))
        plt.subplot(121)
        ls_ = ["-", "--", "-.", ":", "-"]
        for k, i in enumerate(idx):
            if k == 0:
                plt.plot(xi, self.u_xt[:, i], ls_[k], label='$t=%.5f$' % ti[i], lw=2.5)
            else:
                plt.plot(xi, self.u_xt[:, i], ls_[k], label='$t=%.5f$' % ti[i], lw=1.5)
        plt.ylabel('$U(x,t)$', fontdict={"fontsize": 18})
        plt.xlabel('$x$', fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16, loc='best')
        plt.title("波动方程在某些时刻的数值解曲线", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        # 2、带有填充区域的等值线图
        plt.subplot(122)
        extent = [0, self.t_b + self.t_h, 0, self.x_a + self.x_h]  # 时间和空间的取值范围
        plt.contourf(self.u_xt, levels=15, origin='lower', extent=extent, cmap=plt.get_cmap("jet"))
        plt.colorbar()  # 颜色bar
        plt.ylabel('$x$', fontdict={"fontsize": 18})
        plt.xlabel('$t$', fontdict={"fontsize": 18})
        plt.title("波动方程在时刻$t = %.1f$的等值线图" % self.t_b, fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.show()
