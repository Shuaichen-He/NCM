# -*- coding: UTF-8 -*-
"""
@file:bivariate_three_points_lagrange.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class BivariateThreePointsLagrange:
    """
    二元三点拉格朗日插值,基本思想：先固定x对y做一元插值，然后固定y对x做一元插值。
    在矩形网格上的某个小片上做二元三点拉格朗日插值
    """

    def __init__(self, x, y, Z, x0, y0):
        self.Z = np.asarray(Z, dtype=np.float64)
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        if self.Z.shape[0] != len(self.x) and self.Z.shape[1] != len(self.y):
            raise ValueError("插值数据点维度不匹配！")
        if len(x0) == len(y0):
            self.x0 = np.asarray(x0, dtype=np.float64)
            self.y0 = np.asarray(y0, dtype=np.float64)
            self.n0 = len(self.x0)
        else:
            raise ValueError("所求插值数据点(x0, y0)长度不匹配")
        self.n_x, self.n_y = len(self.x), len(self.y)

    def fit_interp_2d(self):
        """
        求解所求插值点的值
        :return:
        """
        Z0 = np.zeros(self.n0)  # 所求插值点的插值
        for k in range(self.n0):
            Z0[k] = self._cal_xy_interp_val_(self.x0[k], self.y0[k])
        return Z0

    def _cal_xy_interp_val_(self, x, y):
        """
        二元三点拉格朗日插值，求解所给插值坐标(x0, y0)的z0值
        :return:
        """
        idx, idy = self.__find_index__(x, y)
        val = 0.0
        # 如下两层循环计算插值
        for i in range(3):  # 0 1 2, 1 2 0, 2, 0, 1
            i1, i2 = np.mod(i + 1, 3), np.mod(i + 2, 3)  # 用于保证i,i1,i2取值不同
            val_x = (x - self.x[idx[i1]]) * (x - self.x[idx[i2]]) / \
                    (self.x[idx[i]] - self.x[idx[i1]]) / \
                    (self.x[idx[i]] - self.x[idx[i2]])  # x轴基函数
            for j in range(3):
                j1, j2 = np.mod(j + 1, 3), np.mod(j + 2, 3)  # 用于保证j,j1,j2取值不同
                val_y = (y - self.y[idy[j1]]) * (y - self.y[idy[j2]]) / \
                        (self.y[idy[j]] - self.y[idy[j1]]) / \
                        (self.y[idy[j]] - self.y[idy[j2]])    # y轴基函数
                # 边界情况处理
                if idx[i] == self.n_x - 1 and idy[j] < self.n_y - 1:
                    val += self.Z[-1, idy[j]] * val_x * val_y
                elif idx[i] < self.n_x - 1 and idy[j] == self.n_y - 1:
                    val += self.Z[idx[i], -1] * val_x * val_y
                elif idx[i] == self.n_x - 1 and idy[j] == self.n_y - 1:
                    val += self.Z[-1, -1] * val_x * val_y
                else:  # 非边界情况
                    val += self.Z[idx[i], idy[j]] * val_x * val_y
        return val

    def __find_index__(self, xi, yi):
        """
        查找坐标值xi、yi所在的区间索引
        :param xi: x轴坐标值
        :param yi: y轴坐标值
        :return:
        """
        idx, idy = np.infty, np.infty  # 初始化x轴和y轴的索引编号
        # 查找所求插值点的区间索引
        for i in range(self.n_x - 1):
            if self.x[i] <= xi <= self.x[i + 1]:
                idx = i
                break
        for i in range(self.n_y - 1):
            if self.y[i] <= yi <= self.y[i + 1]:
                idy = i
                break
        if idx is np.infty or idy is np.infty:
            raise ValueError("所给数据点不能进行外插值！")

        # 针对xi值所在区间最近三个点索引求解
        if idx:  # 所求点x轴不在第一个区间片
            if idx == self.n_x - 2:  # 所求点在最后一个区间片
                near_idx = np.array([self.n_x - 3, self.n_x - 2, self.n_x - 1])
            else:  # 所求点在区间内部片
                if np.abs(self.x[idx - 1] - xi) > np.abs(self.x[idx + 2] - xi):
                    near_idx = np.array([idx, idx + 1, idx + 2])  # 更靠近idx+2
                else:
                    near_idx = np.array([idx - 1, idx, idx + 1])
        else:  # 所求点在第一个区间片
            near_idx = np.array([0, 1, 2])

        # 针对yi值所在区间最近三个点索引求解
        if idy:  # 所求点y轴不在第一个区间片
            if idy == self.n_y - 2:  # 所求点在最后一个区间片
                near_idy = np.array([self.n_y - 3, self.n_y - 2, self.n_y - 1])
            else:  # 所求点在区间内部片
                if np.abs(self.y[idy - 1] - yi) > np.abs(self.y[idy + 2] - yi):
                    near_idy = np.array([idy, idy + 1, idy + 2])
                else:
                    near_idy = np.array([idy - 1, idy, idy + 1])
        else:  # 第一个区间片
            near_idy = np.array([0, 1, 2])
        return near_idx, near_idy

    def plt_3d_surface(self, ax, title, fh=None):
        """
        可视化三维曲面图和等高线图
        :return:
        """
        # 可视化绘图模拟数据插值计算
        n = 200
        x = np.linspace(min(self.x), max(self.x), n)  # 等距划分200份
        y = np.linspace(min(self.y), max(self.y), n)  # 等距划分200份
        xi, yi = np.meshgrid(x, y)  # 生成网格点
        zi = np.zeros((n, n))  # 存储对应网格点的z值
        for i in range(n):
            for j in range(n):
                zi[i, j] = self._cal_xy_interp_val_(xi[i, j], yi[i, j])
        # 可视化三维图像
        ax.plot_surface(xi, yi, zi.T, cmap=plt.get_cmap("rainbow"), rstride=2, cstride=2, lw=1)
        plt.title("二元三点拉格朗日值曲面（%s）" % title, fontdict={"fontsize": 18})
        if fh is not None:
            fz = fh(xi, yi)  # 真值
            mse = np.mean((fz - zi) ** 2)
            plt.title("二元三点拉格朗日（%s），$MSE=%.5e$" % (title, mse), fontdict={"fontsize": 18})
        ax.set_xlabel(r"$x$", fontdict={"fontsize": 18})
        ax.set_ylabel(r"$y$", fontdict={"fontsize": 18})
        ax.set_zlabel(r"$z$", fontdict={"fontsize": 18})
        ax.grid(ls=":")
        # ax.azim = -15
        plt.tick_params(labelsize=16)  # 刻度字体大小16

    def plt_3d_surface_contourf(self, fh=None):
        """
        可视化三维曲面图和等高线图
        :return:
        """
        # 可视化绘图模拟数据插值计算
        n = 200
        x = np.linspace(min(self.x), max(self.x), n)  # 等距划分200份
        y = np.linspace(min(self.y), max(self.y), n)  # 等距划分200份
        xi, yi = np.meshgrid(x, y)  # 生成网格点
        zi = np.zeros((n, n))  # 存储对应网格点的z值
        for i in range(n):
            for j in range(n):
                zi[i, j] = self._cal_xy_interp_val_(xi[i, j], yi[i, j])
        # 可视化三维图像
        fig = plt.figure(figsize=(14, 5))
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(xi, yi, zi.T, cmap=plt.get_cmap("rainbow"), rstride=2, cstride=2, lw=1)
        # ax.contour(xi, yi, zi.T, zdir="z", offset=-150, cmap=plt.get_cmap("coolwarm"))
        plt.title("二元三点拉格朗日插值三维曲面图", fontdict={"fontsize": 18})
        if fh is not None:
            fz = fh(xi, yi)  # 真值
            mse = np.mean((fz - zi) ** 2)
            plt.title("二元三点拉格朗日，$MSE=%.5e$" % mse, fontdict={"fontsize": 18})
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
        plt.title("二元三点拉格朗日插值等值线图", fontdict={"fontsize": 18})
        plt.show()

