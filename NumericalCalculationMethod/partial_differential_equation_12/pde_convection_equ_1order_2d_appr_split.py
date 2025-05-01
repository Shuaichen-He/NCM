# -*- coding: UTF-8 -*-
"""
@file_name: pde_convection_equ_1order_2d_appr_split.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class PDEConvectionEquationFirstOrder2D_ApprSplit:
    """
    双曲型偏微分方程，一阶二维常系数对流方程，近似分裂的差分格式
    不采用三维数组存储，三维数组存储数据量较大，通过指定t_m计算保留到t_m，即保留当前时刻值。
    """

    def __init__(self, a_const, b_const, f_u0, x_span, y_span, t_T, x_n, y_n, t_m):  # 参考LF格式
        # 一阶二维对流方程的常系数
        self.a, self.b = np.asarray(a_const, np.float64), np.asarray(b_const, np.float64)
        self.f_u0 = f_u0  # 边界条件函数，u_0(x,0)
        self.x_a, self.x_b, self.t_T = x_span[0], x_span[1], t_T  # 求解区间，时间默认左端点为0
        self.y_a, self.y_b = y_span[0], y_span[1]  # 求解区间
        self.x_n, self.y_n, self.t_m = x_n, y_n, t_m  # 求解空间和时间划分网格数
        self.u_xyt = None  # 存储pde数值解

    def solve_pde(self):
        """
        求解一阶二维常系数对流方程
        :return:
        """
        x_h, y_h = (self.x_b - self.x_a) / self.x_n, (self.y_b - self.y_a) / self.y_n  # 空间步长
        t_h = self.t_T / self.t_m  # 时间步长
        # 一阶二维对流方程的数值解，此处不用三维数组，而是只保留当前时刻数值解，可为二维数组。
        self.u_xyt = np.zeros((self.x_n + 1 + 4 * self.t_m, self.y_n + 1 + 4 * self.t_m))
        # 二维节点延拓，单独计算
        for i in range(self.x_n + 4 * self.t_m):
            for j in range(self.y_n + 4 * self.t_m):
                xi = self.x_a + (i - 2 * self.t_m) * x_h
                yi = self.y_a + (j - 2 * self.t_m) * y_h
                self.u_xyt[i, j] = self.f_u0(xi, yi)
        # 按照时间维度递推计算
        u_xt_tmp = np.copy(self.u_xyt)  # 临时数组，用于更新递推时间数值解，保留当前时刻值
        r1, r2 = self.a * t_h / x_h / 2, self.b * t_h / y_h / 2  # 差分格式系数
        for k in range(self.t_m):
            for i in range(2 * k + 2, self.x_n + 4 * self.t_m - 2 * k):
                for j in range(2 * k + 1, self.y_n + 4 * self.t_m - 2 * k + 1):
                    u1 = self.u_xyt[i + 1, j] - self.u_xyt[i - 1, j]  # 子项
                    u_xt_tmp[i, j] = self.u_xyt[i, j] - r1 * u1 + 2 * r1 ** 2 * \
                                     (self.u_xyt[i + 1, j] - 2 * self.u_xyt[i, j] +
                                      self.u_xyt[i - 1, j])
            for i in range(2 * k + 2, self.x_n + 4 * self.t_m - 2 * k):
                for j in range(2 * k + 2, self.y_n + 4 * self.t_m - 2 * k):
                    u1 = u_xt_tmp[i, j + 1] - u_xt_tmp[i, j - 1]  # 子项
                    self.u_xyt[i, j] = u_xt_tmp[i, j] - r2 * u1 + 2 * r2 ** 2 * \
                                       (u_xt_tmp[i, j + 1] - 2 * u_xt_tmp[i, j] +
                                        u_xt_tmp[i, j - 1])
        return self.u_xyt

    def plt_convection_surf(self, is_show=True, ax=None):
        """
        可视化指定某时刻t_k的数值解曲面
        :return:
        """
        xi, yi = np.linspace(self.x_a, self.x_b, self.x_n + 1), np.linspace(self.y_a, self.y_b, self.y_n + 1)
        X, Y = np.meshgrid(xi, yi)
        if is_show:
            plt.figure(figsize=(8, 6))
        if ax is None:
            ax = plt.gca(projection='3d')
        ax.plot_surface(X, Y, self.u_xyt[2 * self.t_m:2 * self.t_m + self.x_n + 1,
                              2 * self.t_m:2 * self.t_m + self.y_n + 1], cmap='rainbow')
        ax.set_xlabel("$x$", fontdict={"fontsize": 18})
        ax.set_ylabel("$y$", fontdict={"fontsize": 18})
        ax.set_zlabel("$U$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.title("二维对流方程数值解$(t = %.3f)$" % self.t_T, fontdict={"fontsize": 18})
        if is_show:
            plt.show()
