# -*- coding: UTF-8 -*-
"""
@file_name: pde_laplace_equation_neumann.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt


class PDELaplaceEquationNeumann:
    """
    迭代法求解拉普拉斯方程, Neumann边界条件
    """
    def __init__(self, x_a, y_b, x_h, y_h, eps=1e-3, max_iter=200, pde_model=None,
                 f_ux0=None, f_uxb=None, f_u0y=None, f_uay=None,  # Dirichlet边界条件
                 df_ux0=None, df_uxb=None, df_u0y=None, df_uay=None  # Neumann边界条件
                 ):
        self.f_ux0, self.f_uxb = f_ux0, f_uxb  # 初始边界条件函数，分别表示u(x, 0)和u(x, b)
        self.f_u0y, self.f_uay = f_u0y, f_uay  # 初始边界条件函数，分别表示u(0, y)和u(a, y)
        self.df_ux0, self.df_uxb = df_ux0, df_uxb  # 初始边界条件函数，分别表示u'(x, 0)和u'(x, b)
        self.df_u0y, self.df_uay = df_u0y, df_uay  # 初始边界条件函数，分别表示u'(0, y)和u'(a, y)
        self.x_a, self.y_b = x_a, y_b  # 分别表示自变量x和y的求解区域右端点
        self.x_h, self.y_h = x_h, y_h  # 分别表示自变量x和y的求解步长
        self.n, self.m = int(self.x_a / self.x_h) + 1, int(self.y_b / self.y_h) + 1  # 划分网格区间点数
        self.u_xt = None  # 存储pde数值解
        self.pde_model = pde_model  # 解析解存在的情况下，可进行误差分析
        self.eps, self.max_iter = eps, max_iter  # 迭代法求解精度和最大迭代次数

    def boundary_condition(self):
        """
        边界条件
        :return:
        """
        ave = (self.x_a * (self.f_ux0(0) + self.f_uxb(0)) +
               self.y_b * (self.f_u0y(0) + self.f_uay(0))) / (2 * (self.x_a + self.y_b))
        self.u_xt = ave * np.ones((self.n, self.m))  # 初始数值解
        # Dirichlet边界条件
        i, j = np.arange(0, self.n), np.arange(0, self.m)  # 离散点索引
        if self.f_u0y:
            self.u_xt[0, :] = self.f_u0y(j * self.y_h)  # 左边
            self.u_xt[0, 0] = (self.u_xt[0, 1] + self.u_xt[1, 0]) / 2  # 左下角点
        if self.f_uay:
            self.u_xt[-1, :] = self.f_uay(j * self.y_h)  # 右边
            self.u_xt[-1, -1] = (self.u_xt[-2, -1] + self.u_xt[-1, -2]) / 2  # 右上角点
        if self.f_ux0:
            self.u_xt[:, 0] = self.f_ux0(i * self.x_h)  # 底部
            self.u_xt[-1, 0] = (self.u_xt[-2, 0] + self.u_xt[-1, 1]) / 2  # 右下角点
        if self.f_uxb:
            self.u_xt[:, -1] = self.f_uxb(i * self.x_h)  # 顶部
            self.u_xt[0, -1] = (self.u_xt[0, -2] + self.u_xt[1, -1]) / 2  # 左上角点

        # Neumann边界条件
        if self.df_u0y:  # 左边界
            self.u_xt[0, 1:-1] = (2 * self.y_h * self.df_u0y(j[1:-1] * self.y_h) + 2 * self.u_xt[1, 1:-1] +
                                  self.u_xt[0, :-2] + self.u_xt[0, 2:]) / 4
            # 左下顶点
            self.u_xt[0, 0] = (2 * self.y_h * self.df_u0y(0) + self.u_xt[1, 0] + self.u_xt[0, 1]) / 2
        if self.df_uay:  # 右边界
            self.u_xt[-1, 1:-1] = (2 * self.y_h * self.df_uay(j[1:-1] * self.y_h) + 2 * self.u_xt[-2, 1:-1] +
                                   self.u_xt[-1, :-2] + self.u_xt[-1, 2:]) / 4
            # 右下顶点
            self.u_xt[-1, 0] = (2 * self.y_h * self.df_uay(self.y_b) + self.u_xt[-2, 0] + self.u_xt[-1, 1]) / 2
        if self.df_ux0:  # 在底部
            self.u_xt[1:-1, 0] = (2 * self.x_h * self.df_ux0(i[1:-1] * self.x_h) + 2 * self.u_xt[1:-1, 1] +
                                  self.u_xt[:-2, 0] + self.u_xt[2:, 0]) / 4
            # 左上顶点
            self.u_xt[0, -1] = (2 * self.x_h * self.df_ux0(0) + self.u_xt[0, -2] + self.u_xt[1, -1]) / 2
        if self.df_uxb:  # 在顶部
            self.u_xt[1:-1, -1] = (2 * self.x_h * self.df_uxb(i[1:-1] * self.x_h) + 2 * self.u_xt[1:-1, -2] +
                                   self.u_xt[:-2, -1] + self.u_xt[2:, -1]) / 4
            # 右上顶点
            self.u_xt[-1, -1] = (2 * self.x_h * self.df_uxb(self.x_a) + self.u_xt[-2, -1] + self.u_xt[-1, -2]) / 2


    def solve_pde(self):
        """
        迭代法求解拉普拉斯方程
        :return:
        """
        self.boundary_condition()  # 边界条件计算
        # 松弛因子计算
        w = 4 / (2 + np.sqrt(4 - (np.cos(np.pi / (self.n - 1)) + np.cos(np.pi / (self.m - 1))) ** 2))
        # 松弛迭代法求解
        err, iter_ = 1, 0  # 初始化误差和迭代次数
        while err > self.eps and iter_ < self.max_iter:
            print(iter_)
            err, iter_ = 0.0, iter_ + 1
            for j in range(1, self.m - 1):
                if self.df_u0y:  # 左边界
                    self.u_xt[0, j] += w * (2 * self.y_h * self.df_u0y(j * self.y_h) + 2 * self.u_xt[1, j] +
                                            self.u_xt[0, j - 1] + self.u_xt[0, j + 1] - 4 * self.u_xt[0, j]) / 4
                if self.df_uay:  # 右边界
                    self.u_xt[-1, j] += w * (2 * self.y_h * self.df_uay(j * self.y_h) + 2 * self.u_xt[-2, j] +
                                             self.u_xt[-1, j - 1] + self.u_xt[-1, j + 1] - 4 * self.u_xt[-1, j]) / 4
                for i in range(1, self.n - 1):
                    relax = w * (self.u_xt[i, j + 1] + self.u_xt[i, j - 1] + self.u_xt[i + 1, j] +
                                 self.u_xt[i - 1, j] - 4 * self.u_xt[i, j]) / 4
                    self.u_xt[i, j] += relax
                    # 边界情况迭代
                    if self.df_ux0:  # 在底部
                        self.u_xt[i, 0] += w * (2 * self.x_h * self.df_ux0(i * self.x_h) +
                                                2 * self.u_xt[i, 1] + self.u_xt[i - 1, 0] +
                                                self.u_xt[i + 1, 0] - 4 * self.u_xt[i, 0]) / 4
                    if self.df_uxb:  # 在顶部
                        self.u_xt[i, -1] += w * (2 * self.x_h * self.df_uxb(i * self.x_h) +
                                                 2 * self.u_xt[i, -2] + self.u_xt[i - 1, -1] +
                                                 self.u_xt[i + 1, -1] - 4 * self.u_xt[i, -1]) / 4
                    if err <= abs(relax):
                        err = abs(relax)
        print("iter_num = %d，max r_ij = %.10e" % (iter_, err))
        print(self.u_xt.T)
        return self.u_xt.T

    def plt_pde_laplace_surface(self):  # 参考一维波动方程
        """
        可视化数值解
        :return:
        """
        xi = np.linspace(0, self.x_a, self.n)
        yi = np.linspace(0, self.y_b, self.m)
        x, y = np.meshgrid(xi, yi)
        if self.pde_model:
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(121, projection='3d')
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca(projection='3d')
        ax.plot_surface(x, y, self.u_xt.T, cmap='rainbow')
        ax.set_xlabel("x", fontdict={"fontsize": 14})
        ax.set_ylabel("y", fontdict={"fontsize": 14})
        ax.set_zlabel("U", fontdict={"fontsize": 14})
        plt.title("Laplace equation numerical solution", fontdict={"fontsize": 16})
        if self.pde_model:
            ax = fig.add_subplot(122, projection='3d')
            analytical_sol = self.pde_model(x, y)
            error_ = analytical_sol - self.u_xt.T  # 误差
            print("均方误差：", np.sqrt(np.mean(error_ ** 2)))
            # 不考虑边界值
            ax.plot_surface(x[1:-1, 1:-1], y[1:-1, 1:-1], error_[1:-1, 1:-1], cmap='rainbow')
            z_format = plt.FormatStrFormatter('%.e')  # 设置y轴标签文本的格式
            ax.zaxis.set_major_formatter(z_format)
            ax.set_xlabel("x", fontdict={"fontsize": 14})
            ax.set_ylabel("y", fontdict={"fontsize": 14})
            ax.set_zlabel("Error", fontdict={"fontsize": 14})
            plt.title("The error of laplace numerical solution", fontdict={"fontsize": 16})
            # fig.tight_layout()
        plt.show()

    def plt_pde_laplace_curve_contourf(self):  # 参考一维波动方程
        """
        可视化某些时刻的数值解，以及等值线图
        :return:
        """
        # 1、不同时刻的波的传播随空间坐标的变化
        xi = np.linspace(0, self.x_a, self.n)
        yi = np.linspace(0, self.y_b, self.m)
        idx = np.array([1, len(yi) / 4, len(yi) / 2, 3 * len(yi) / 4, len(yi)], np.int) - 1
        plt.figure(figsize=(14, 5))
        plt.subplot(121)
        for i in idx:
            plt.plot(xi, self.u_xt[:, i], label='y=%.5f' % yi[i], lw=1.5)
        plt.ylabel('U(x,y)', fontdict={"fontsize": 12})
        plt.xlabel('x', fontdict={"fontsize": 12})
        plt.legend(loc='upper right')
        plt.title("Numerical solution of Laplace at some y values", fontdict={"fontsize": 14})
        # 2、带有填充区域的等值线图
        plt.subplot(122)
        extent = [0, self.y_b + self.y_h, 0, self.x_a + self.x_h]  # 时间和空间的取值范围
        plt.contourf(self.u_xt, levels=20, origin='lower', extent=extent, cmap=plt.get_cmap("jet"))
        # plt.clabel(c, inline=True, fontsize=10)  # 添加等高数值标记
        plt.colorbar()  # 颜色bar
        plt.ylabel('y', fontdict={"fontsize": 12})
        plt.xlabel('x', fontdict={"fontsize": 12})
        plt.title("The contour face of Laplace equation", fontdict={"fontsize": 14})
        plt.show()
