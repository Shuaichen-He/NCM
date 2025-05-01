# -*- coding: UTF-8 -*-
"""
@file:slice_bilinear_interpolation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class SliceBiLinearInterpolation:
    """
    分片双线性插值，每个网格拟合一个二次曲面
    """

    def __init__(self, x, y, Z, x0, y0):
        self.Z = np.asarray(Z, dtype=np.float64)  # 转化为ndarray，方便数值计算
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        if self.Z.shape[0] != len(self.x) and self.Z.shape[1] != len(self.y):
            raise ValueError("插值数据点维度不匹配！")
        if len(x0) == len(y0):
            self.x0 = np.asarray(x0, dtype=np.float64)  # 待求插值点
            self.y0 = np.asarray(y0, dtype=np.float64)
            self.n0 = len(self.x0)
        else:
            raise ValueError("所求插值数据点(x0, y0)长度不匹配")
        self.n_x, self.n_y = len(self.x), len(self.y)
        self.Z0 = None  # 所求插值点的值

    def fit_2d_interp(self):
        """
        求解所求插值点的值
        :return:
        """
        self.Z0 = np.zeros(self.n0)  # 所求插值点的插值初始化
        for k in range(self.n0):  # 求解每个插值点的Z0
            Lxy = self._fit_bi_linear_(self.x0[k], self.y0[k])
            v_ = np.array([1, self.x0[k], self.y0[k], self.x0[k] * self.y0[k]])
            self.Z0[k] = np.dot(v_, Lxy)
        return self.Z0

    def _fit_bi_linear_(self, x, y):
        """
        分片双线性插值，求解所给点的多项式系数
        :return:
        """
        idx, idy = self.__find_index__(x, y)  # 查找插值点所在的矩形网格上的某个小片索引
        x_1i, x_i = self.x[idx], self.x[idx + 1]  # x片值
        y_li, y_i = self.y[idy], self.y[idy + 1]  # y片值
        # 构造矩阵求解a,b,c,d
        node_mat = np.array([[1, x_1i, y_li, x_1i * y_li], [1, x_i, y_i, x_i * y_i],
                             [1, x_1i, y_i, x_1i * y_i], [1, x_i, y_li, x_i * y_li]])
        vector_z = np.array([self.Z[idx, idy], self.Z[idx + 1, idy + 1],
                             self.Z[idx, idy + 1], self.Z[idx + 1, idy]])
        coefficient = np.linalg.solve(node_mat, vector_z)
        return coefficient

    def __find_index__(self, xi, yi):
        """
        查找坐标值xi、yi所在的区间索引
        :param xi: x轴坐标值
        :param yi: y轴坐标值
        :return:
        """
        idx, idy = np.infty, np.infty
        # 查找所求插值点的区间索引
        for i in range(self.n_x - 1):
            if self.x[i] <= xi <= self.x[i + 1] or self.x[i + 1] <= xi <= self.x[i]:
                idx = i
                break
        for j in range(self.n_y - 1):
            if self.y[j] <= yi <= self.y[j + 1] or self.y[j + 1] <= yi <= self.y[j]:
                idy = j
                break
        if idx is np.infty or idy is np.infty:
            raise ValueError("所给数据点不能进行外插值！")
        return idx, idy

    def plt_3d_surface(self, ax, title, fh=None):
        """
        可视化三维曲面图和等高线图
        :return:
        """

        # 定义子函数
        def __cal_xy_plt__(x_i, y_i):
            """
            求解所求区间模拟等分点的值，用于绘图
            :return:
            """
            lxy = self._fit_bi_linear_(x_i, y_i)
            v_ = np.array([1, x_i, y_i, x_i * y_i])
            z_i = np.dot(v_, lxy)
            return z_i

        # 可视化绘图模拟数据插值计算
        n = 50
        x = np.linspace(min(self.x), max(self.x), n)  # 等距划分200份
        y = np.linspace(min(self.y), max(self.y), n)  # 等距划分200份
        xi, yi = np.meshgrid(x, y)  # 生成网格点
        zi = np.zeros((n, n))  # 存储对应网格点的z值
        for i in range(n):
            for j in range(n):
                zi[i, j] = __cal_xy_plt__(xi[i, j], yi[i, j])
        # 可视化三维图像
        ax.plot_surface(xi, yi, zi.T, cmap=plt.get_cmap("rainbow"), rstride=2, cstride=2, lw=1)
        plt.title("插值曲面（%s）" % title, fontdict={"fontsize": 18})
        if fh is not None:
            fz = fh(xi, yi)  # 真值
            mse = np.mean((fz - zi) ** 2)
            plt.title("%s，$MSE=%.5e$" % (title, mse), fontdict={"fontsize": 18})
        ax.set_xlabel(r"$x$", fontdict={"fontsize": 18})
        ax.set_ylabel(r"$y$", fontdict={"fontsize": 18})
        ax.set_zlabel(r"$z$", fontdict={"fontsize": 18})
        ax.grid(ls=":")
        # ax.azim = -15
        plt.tick_params(labelsize=16)  # 刻度字体大小16

    def plt_3d_surface_contourf(self):
        """
        可视化三维曲面图和等高线图
        :return:
        """

        # 定义子函数
        def __cal_xy_plt__(x_i, y_i):
            """
            求解所求区间模拟等分点的值，用于绘图
            :return:
            """
            lxy = self._fit_bi_linear_(x_i, y_i)
            v_ = np.array([1, x_i, y_i, x_i * y_i])
            z_i = np.dot(v_, lxy)
            return z_i

        # 可视化绘图模拟数据插值计算
        n = 200
        x = np.linspace(min(self.x), max(self.x), n)  # 等距划分200份
        y = np.linspace(min(self.y), max(self.y), n)  # 等距划分200份
        xi, yi = np.meshgrid(x, y)  # 生成网格点
        zi = np.zeros((n, n))  # 存储对应网格点的z值
        for i in range(n):
            for j in range(n):
                zi[i, j] = __cal_xy_plt__(xi[i, j], yi[i, j])
        # 可视化三维图像
        fig = plt.figure(figsize=(14, 5))
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(xi, yi, zi.T, cmap=plt.get_cmap("rainbow"), rstride=2, cstride=2, lw=1)
        # ax.contour(xi, yi, zi.T, zdir="z", offset=-150, cmap=plt.get_cmap("coolwarm"))
        plt.title("双线性插值三维曲面图", fontdict={"fontsize": 18})
        ax.set_xlabel(r"$x$", fontdict={"fontsize": 18})
        ax.set_ylabel(r"$y$", fontdict={"fontsize": 18})
        ax.set_zlabel(r"$z$", fontdict={"fontsize": 18})
        ax.grid(ls=":")
        # ax.azim = -15
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        # 绘制等高线图
        plt.subplot(122)
        plt.contourf(xi, yi, zi, 15, cmap=plt.get_cmap("rainbow"))
        plt.xlabel(r"$x$", fontdict={"fontsize": 18})
        plt.ylabel(r"$y$", fontdict={"fontsize": 18})
        cb = plt.colorbar()  # 添加颜色条
        cb.ax.tick_params(labelsize=16)  # 颜色条标记大小
        plt.tick_params(labelsize=16)  # 刻度字体大小14
        plt.title("双线性插值等值线图", fontdict={"fontsize": 18})
        plt.show()
