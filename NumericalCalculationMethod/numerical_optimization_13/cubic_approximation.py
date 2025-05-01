# -*- coding: UTF-8 -*-
"""
@file_name: cubic_approximation.py
@time: 2022-09-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from util_font import *


class CubicApproximationOptimization:
    """
    三次逼近方法，求解单变量函数的极值问题。
    """

    def __init__(self, fun, x_span, eps, max_iter=1000, is_minimum=True):
        t = fun.free_symbols.pop()  # 获取优化目标函数的自由符号变量
        self.fun = sympy.lambdify(t, fun, "numpy")  # 优化函数，转化为lambda函数
        self.d_fun = sympy.lambdify(t, sympy.diff(fun), "numpy")  # 一阶导函数
        self.a, self.b = x_span[0], x_span[1]  # 单峰区间
        self.eps = eps  # 精度要求
        self.max_iter = max_iter  # 最大迭代次数
        self.is_minimum = is_minimum  # 是否是极小值，极大值设置为False
        self.local_extremum = None  # 搜索过程，极值点

    def fit_optimize(self):
        """
        3次逼近方法，求解算法
        :return:
        """
        h = 2 * (self.b - self.a) / self.d_fun(self.a)  # 初始h
        p0, p1 = self.a, self.a + h  # 第一个和第二个点
        local_extremum = [[p0, self.fun(p0)]]
        d_p0 = self.d_fun(p0)  # 一阶导函数在p0点的值
        while np.abs(d_p0) > self.eps:
            F, G = self.fun(p1) - self.fun(p0), h * (self.d_fun(p1) - d_p0)
            alpha = G - 2 * (F - d_p0 * h)
            gamma_tmp = G - 3 * alpha + \
                        np.sqrt((G - 3 * alpha) ** 2 - 12 * alpha * h * d_p0)
            if np.abs(gamma_tmp) < 1e-51:  # 避免分母过小
                break
            gamma = -2 * h * d_p0 / gamma_tmp
            p2 = p0 + gamma * h  # 第三个点
            h = p2 - p1  # 新的步长
            p0, p1 = p1, p2  # p1替换p0，p2替换p1，继续求解p2
            d_p0 = self.d_fun(p0)  # 一阶导函数在p0点的值
            local_extremum.append([p0, self.fun(p0)])
        self.local_extremum = np.asarray(local_extremum)
        if self.is_minimum is False:  # 极大值
            self.local_extremum[:, 1] = -1 * self.local_extremum[:, 1]
        return self.local_extremum[-1]

    def plt_optimization(self, plt_zone=None):
        """
        可视化优化过程
        :param plt_zone:  可视化的区间
        :return:
        """
        if plt_zone is not None:
            xi = np.linspace(plt_zone[0], plt_zone[1], 150)
        else:
            xi = np.linspace(self.a, self.b, 150)
        plt.figure(figsize=(14, 5))
        plt.subplot(121)
        if self.is_minimum:
            plt.plot(xi, self.fun(xi), "k-", lw=1.5, label="$f(x)$")
        else:
            plt.plot(xi, -1 * self.fun(xi), "k-", lw=1.5, label="$f(x)$")
        plt.plot(self.local_extremum[-1, 0], self.local_extremum[-1, 1], "ro", label="$(x^*, f(x^*))$")
        plt.xlabel("$x$", fontdict={"fontsize": 18})
        plt.ylabel("$f(x)$", fontdict={"fontsize": 18})
        plt.title("函数局部极值点$(%.10f, %.10f)$"
                  % (self.local_extremum[-1, 0], self.local_extremum[-1, 1]), fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16)
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        plt.subplot(122)
        plt.plot(np.arange(1, len(self.local_extremum) + 1), self.local_extremum[:, 1], "o--")
        plt.xlabel("搜索次数", fontdict={"fontsize": 18})
        plt.ylabel("$f(x^*)$", fontdict={"fontsize": 18})
        plt.title("函数极值优化过程，三次逼近法搜索$%d$次" % len(self.local_extremum), fontdict={"fontsize": 18})
        # plt.legend(frameon=False, fontsize=16)
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        plt.show()
