# -*- coding: UTF-8 -*-
"""
@file_name: golden_section_search.py
@time: 2022-09-09
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import math
from util_font import *


class GoldenSectionSearchOptimization:
    """
    黄金分割搜索法，求解单变量函数的极值问题
    """

    def __init__(self, fun, x_span, eps, is_minimum=True):
        self.fun = fun  # 优化函数
        self.a, self.b = x_span[0], x_span[1]  # 单峰区间
        self.eps = eps  # 精度要求
        self.is_minimum = is_minimum  # 是否是极小值，极大值设置为False
        self.r = (-1 + math.sqrt(5)) / 2  # 黄金分割比例
        self.local_extremum = None  # 搜索过程，极值点
        self.reduce_zone = []  # 搜索过程，缩小的区间

    def fit_optimize(self):
        """
        黄金分割搜索优化算法
        :return:
        """
        a, b = self.a, self.b  # 区间端点
        local_extremum = []  # 搜索过程，极值点
        c, d = a + (1 - self.r) * (b - a), b - (1 - self.r) * (b - a)  # 内点
        tol, k = np.abs(d - c), 1  # 精度与迭代次数
        while tol > self.eps:
            fc, fd = self.fun(c), self.fun(d)  # 函数值的更新
            if self.is_minimum:  # 极小值
                if fc > fd:
                    local_extremum.append([d, fd])  # 存储当前极值
                    a, c = c, d  # 区间为[c, d]
                    d = b - (1 - self.r) * (b - a)  # 内点更新
                else:
                    local_extremum.append([c, fc])  # 存储当前极值
                    b, d = d, c  # 区间为[a, d]
                    c = a + (1 - self.r) * (b - a)  # 内点更新
            else:  # 极大值
                if fc < fd:
                    local_extremum.append([d, fd])
                    a, c = c, d
                    d = b - (1 - self.r) * (b - a)
                else:
                    local_extremum.append([c, fc])
                    b, d = d, c
                    c = a + (1 - self.r) * (b - a)
            self.reduce_zone.append([c, d])  # 缩小的区间
            tol, k = np.abs(d - c), k + 1  # 更新精度和迭代次数
        self.local_extremum = np.asarray(local_extremum)
        return self.local_extremum[-1]

    def plt_optimization(self, plt_zone=None):
        """
        可视化优化过程
        :param plt_zone:  可视化的区间
        :return:
        """
        self.reduce_zone = np.asarray(self.reduce_zone)
        if plt_zone is not None:
            xi = np.linspace(plt_zone[0], plt_zone[1], 150)
        else:
            xi = np.linspace(self.a, self.b, 150)
        plt.figure(figsize=(14, 5))
        plt.subplot(121)
        plt.plot(xi, self.fun(xi), "k-", lw=1.5, label="$f(x)$")
        plt.plot(self.local_extremum[-1, 0], self.local_extremum[-1, 1], "ro", label="$(x^*, f(x^*))$")
        plt.xlabel("$x$", fontdict={"fontsize": 18})
        plt.ylabel("$f(x)$", fontdict={"fontsize": 18})
        if self.is_minimum:
            plt.title("函数局部极小值点 $(%.10f, %.10f)$"
                      % (self.local_extremum[-1, 0], self.local_extremum[-1, 1]), fontdict={"fontsize": 18})
        else:
            plt.title("函数局部极大值点 $(%.10f, %.10f)$"
                      % (self.local_extremum[-1, 0], self.local_extremum[-1, 1]), fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16)
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        plt.subplot(122)
        plt.semilogy(np.arange(1, len(self.local_extremum) + 1),
                     (self.reduce_zone[:, 1] - self.reduce_zone[:, 0]), "*-",
                     label="$d - c = %.10e$" % (self.reduce_zone[-1, 1] - self.reduce_zone[-1, 0]))
        plt.xlabel("搜索次数$k$", fontdict={"fontsize": 18})
        plt.ylabel("$\epsilon=d_k - c_k$", fontdict={"fontsize": 18})
        plt.title("内点区间$[c, d]$压缩过程，黄金分割搜索$k=%d$次" % len(self.local_extremum), fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16)
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        plt.show()
