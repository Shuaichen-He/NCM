# -*- coding: UTF-8 -*-
"""
@file_name: nelder_mead_2d.py
@time: 2022-09-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class NelderMeadOptimization:
    """
    內德-米德优化算法，仅针对2元函数
    """

    def __init__(self, fun, V_k, eps, is_minimum=True):
        self.fun = fun  # 优化函数，2元
        self.V_k = np.asarray(V_k)  # 初始的三个顶点，格式3 * 3
        S = np.hstack([self.V_k, np.ones((self.V_k.shape[0], 1))])
        if np.abs(np.linalg.det(S)) < 1e-2:
            raise ValueError("三点近似共线，请重新选择顶点。")
        self.eps = eps  # 精度要求
        self.is_minimum = is_minimum  # 是否是极小值，极大值设置为False
        self.local_extremum = None  # 搜索过程，极值点

    def fit_optimize(self):
        """
        內德-米德优化二元函数算法的核心内容
        :return:
        """
        m, n = self.V_k.shape
        local_extremum = []  # 存储迭代过程中的极值点
        epsilon = 1  # 初始精度
        f_val = np.zeros(m)  # 顶点的函数值
        while epsilon > self.eps:
            # 对三个顶点排序
            for i in range(m):  # 计算三个顶点的函数值
                f_val[i] = self.fun(self.V_k[i, :])
            idx = np.argsort(f_val)  # 函数值从小到大排序，获得排序索引
            B, G, W = self.V_k[idx]  # 对坐标点排序，B为最优，G次之，W最差
            f_val = f_val[idx]  # 对函数值排序
            M = (B + G) / 2  # BG线段的中点
            R = 2 * M - W  # 计算反射点
            f_R = self.fun(R)  # 反射点的函数值
            if f_R < f_val[1]:  # f(R) < f(G)，情况1
                if f_val[0] < f_R:  # f(B) < f(R)，反射点值最小
                    W = R  # 更新最差顶点
                else:  # 从优到差为B、R、G、W
                    E = 2 * R - M  # 开拓点
                    # 开拓点E比B优，则更新最差顶点，否则，以反射点更新
                    W = E if self.fun(E) < f_val[0] else R
            else:  # 情况2，需考虑收缩和向B方向收缩
                if f_R < f_val[-1]:  # 比最差的要好
                    W = R  # 更新
                C1, C2 = (W + M) / 2, (M + R) / 2  # 计算两个收缩点
                f_C1, f_C2 = self.fun(C1), self.fun(C2)  # 收缩点对应的函数值
                [C, f_C] = [C1, f_C1] if f_C1 < f_C2 else [C2, f_C2]  # 选择最优的收缩点
                if f_C < f_val[-1]:  # 收缩点C比W要好
                    W = C  # 更新
                else:
                    S = (B + W) / 2  # 向B方向收缩
                    W, G = S, M  # 以线段BW、BG中点更新顶点W和G
            self.V_k = np.array([W, B, G])  # 组合顶点，形成三角形
            local_extremum.append([W[0], W[1], self.fun(W)])  # 存储最优的
            # 以顶点G和W与B顶点函数值的距离模为精度判断标准
            epsilon = np.linalg.norm([f_val[0] - f_val[1], f_val[0] - f_val[2]])
        print(epsilon)
        self.local_extremum = np.asarray(local_extremum)
        if self.is_minimum is False:  # 极大值
            self.local_extremum[:, -1] = -1 * self.local_extremum[:, -1]
        return self.local_extremum[-1]

    def plt_optimization(self, x_zone, y_zone):
        """
        可视化优化过程
        :param x_zone:  可视化x坐标的区间
        :param y_zone:  可视化y坐标的区间
        :return:
        """
        e_p = self.local_extremum[-1]  # 极值点
        xi, yi = np.linspace(x_zone[0], x_zone[1], 100), np.linspace(y_zone[0], y_zone[1], 100)
        x, y = np.meshgrid(xi, yi)
        fxy = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                fxy[i, j] = self.fun([x[i, j], y[i, j]])
        fig = plt.figure(figsize=(14, 5))
        ax = fig.add_subplot(121)
        if self.is_minimum:
            c = plt.contour(x, y, fxy, levels=15, cmap=plt.get_cmap("jet"))
            plt.clabel(c, inline=True, fontsize=10)  # 添加等高数值标记
            plt.plot(e_p[0], e_p[1], "ko")
        else:
            c = plt.contour(x, y, -1 * fxy, levels=15, cmap=plt.get_cmap("jet"))
            plt.clabel(c, inline=True, fontsize=10)  # 添加等高数值标记
            plt.plot(e_p[0], e_p[1], "ko")
        ax.set_xlabel("$x$", fontdict={"fontsize": 18})
        ax.set_ylabel("$y$", fontdict={"fontsize": 18})
        plt.title("函数局部极值点$((%.5f, %.5f), %.5f)$" % (e_p[0], e_p[1], e_p[2]), fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        plt.subplot(122)
        plt.plot(np.arange(1, len(self.local_extremum) + 1), self.local_extremum[:, -1], "k*--",
                 markerfacecolor="r", markeredgecolor="r")
        plt.xlabel("搜索次数", fontdict={"fontsize": 18})
        plt.ylabel("$f(x^*, y^*)$", fontdict={"fontsize": 18})
        plt.title("$Nelder-Mead$优化过程，迭代$%d$次" % len(self.local_extremum), fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        plt.show()
