# -*- coding: UTF-8 -*-
"""
@file_name: pde_convection_equation_1order_2d.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class PDEConvectionEquationFirstOrder2D_LF:
    """
    双曲型偏微分方程，一阶二维常系数对流方程，拉克斯-弗里德里希斯（lax_friedrichs）差分格式
    注意：可不采用三维数组存储，三维数组存储数据量较大，
    可通过指定t_m计算保留到t_m，即保留当前时刻值。
    """

    def __init__(self, a_const, b_const, f_u0, x_span, y_span, t_T, x_n, y_n, t_m):
        self.a, self.b = a_const, b_const  # 一阶二维对流方程的常系数
        self.f_u0 = f_u0  # 边界条件函数，u_0(x,y,0)
        self.x_a, self.x_b, self.t_T = x_span[0], x_span[1], t_T  # 求解区间，时间默认左端点为0
        self.y_a, self.y_b = y_span[0], y_span[1]  # 求解区间
        self.x_n, self.y_n, self.t_m = x_n, y_n, t_m  # 求解空间和时间划分数
        self.u_xyt = None  # 存储pde数值解

    def solve_pde(self):
        """
        求解一阶二维常系数对流方程
        :return:
        """
        x_h, y_h = (self.x_b - self.x_a) / self.x_n, (self.y_b - self.y_a) / self.y_n  # 空间步长
        t_h = self.t_T / self.t_m  # 时间步长
        # 一阶二维对流方程的数值解，注意此处可不用三维数组，
        # 而是只保留当前时刻数值解，可为二维数组。
        self.u_xyt = np.zeros((self.t_m + 1, self.x_n + 1 + 2 * self.t_m,
                               self.y_n + 1 + 2 * self.t_m))
        r1, r2 = abs(self.a) * t_h / x_h / 2, abs(self.b) * t_h / y_h / 2  # 差分格式系数
        if r1 ** 2 + r2 ** 2 > 0.5:
            raise ValueError("Lax_Friedrichs差分格式不稳定")
        else:
            print("Lax_Friedrichs稳定条件：%.5f" % (r1 ** 2 + r2 ** 2))
        # 二维节点延拓，单独计算
        for i in range(self.x_n + 2 * self.t_m):
            for j in range(self.y_n + 2 * self.t_m):
                x_i = self.x_a + (i - self.t_m) * x_h
                y_i = self.y_a + (j - self.t_m) * y_h
                self.u_xyt[0, i, j] = self.f_u0(x_i, y_i)
        # 按照时间维度递推计算
        for k in range(self.t_m):
            for i in range(k + 1, self.x_n + 2 * self.t_m - k):
                for j in range(k + 1, self.y_n + 2 * self.t_m - k):
                    term1 = (self.u_xyt[k, [i + 1, i - 1], j].sum() +
                             self.u_xyt[k, i, [j + 1, j - 1]].sum()) / 4
                    term2 = r1 * (self.u_xyt[k, i + 1, j] - self.u_xyt[k, i - 1, j])
                    term3 = r2 * (self.u_xyt[k, i, j + 1] - self.u_xyt[k, i, j - 1])
                    self.u_xyt[k + 1, i, j] = term1 - term2 - term3
        return self.u_xyt

    def plt_convection_surf(self, t_k, is_show=True, ax=None):
        """
        可视化指定某时刻t_k的数值解曲面
        :return:
        """
        xi = np.linspace(self.x_a, self.x_b, self.x_n + 1)
        yi = np.linspace(self.y_a, self.y_b, self.y_n + 1)
        ti = np.linspace(0, self.t_T, self.t_m + 1)
        idx = 0  # 查找所指定时刻所在的索引下标
        for i in range(2, len(ti)):
            if ti[i - 1] < t_k <= ti[i]:
                idx = i
                break
        X, Y = np.meshgrid(xi, yi)
        if is_show:
            plt.figure(figsize=(7, 5))
        if ax is None:
            ax = plt.gca(projection='3d')
        ax.plot_surface(X, Y, self.u_xyt[idx, self.t_m:self.t_m + self.x_n + 1,
                              self.t_m:self.t_m + self.y_n + 1], cmap='rainbow')
        ax.set_xlabel("$x$", fontdict={"fontsize": 18})
        ax.set_ylabel("$y$", fontdict={"fontsize": 18})
        ax.set_zlabel("$U$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.title("二维对流方程数值解$(t = %.3f)$" % t_k, fontdict={"fontsize": 18})
        if is_show:
            plt.show()
