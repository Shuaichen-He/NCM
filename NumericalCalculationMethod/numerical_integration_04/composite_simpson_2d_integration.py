# -*- coding: UTF-8 -*-
"""
@file:composite_simpson_2d_integration.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import numpy as np
from util_font import *


class CompositeSimpsonDoubleIntegration:
    """
    复化辛普生二重积分：每次划分区间数递增，对比两次积分精度，满足精度即可。
    """

    int_value = None  # 最终积分值
    sub_interval_num = 0  # 子区间划分数

    def __init__(self, int_fun, x_span, y_span, eps=1e-6, max_split=100, increment=10):
        self.int_fun = int_fun  # 被积函数
        self.x_span = np.asarray(x_span, dtype=np.float64)  # x积分区间
        self.y_span = np.asarray(y_span, dtype=np.float64)  # y积分区间
        self.eps = eps  # 积分精度，为前后两次区间划分积分值的变化
        self.max_split = max_split  # 最大划分次数，每次递增10，可划分的最大区间数10*100
        self.increment = increment  # 默认划分区间数为10，增量为10
        self._integral_values = []  # 存储每次积分值
        self._n_splits = []  # 存储每次划分区间数

    def fit_2d_int(self):
        """
        二重数值积分
        :return:
        """
        int_val, n = 0, 0
        for i in range(self.max_split):
            n = self.increment * (i + 1)  # 划分区间数
            hx, hy = np.diff(self.x_span) / n, np.diff(self.y_span) / n  # 区间步长
            # x和y划分节点
            xi = np.linspace(self.x_span[0], self.x_span[1], n + 1, endpoint=True)
            yi = np.linspace(self.y_span[0], self.y_span[1], n + 1, endpoint=True)
            xy = np.meshgrid(xi, yi)
            int1 = np.sum(self.int_fun(xy[0][:-1, :-1], xy[1][:-1, :-1]))
            int2 = np.sum(self.int_fun(xy[0][1:, 1:], xy[1][:-1, :-1]))
            int3 = np.sum(self.int_fun(xy[0][:-1, :-1], xy[1][1:, 1:]))
            int4 = np.sum(self.int_fun(xy[0][1:, 1:], xy[1][1:, 1:]))
            xci = np.divide(xy[0][:-1, :-1] + xy[0][1:, 1:], 2)  # x各节点中点
            yci = np.divide(xy[1][:-1, :-1] + xy[1][1:, 1:], 2)  # y各节点中点
            int5 = np.sum(self.int_fun(xci, xy[1][:-1, :-1])) + \
                   np.sum(self.int_fun(xy[0][:-1, :-1], yci)) + \
                   np.sum(self.int_fun(xy[0][1:, 1:], yci)) + \
                   np.sum(self.int_fun(xci, xy[1][1:, 1:]))
            int6 = np.sum(self.int_fun(xci, yci))
            int_val = hx * hy / 36 * (int1 + int2 + int3 + int4 + 4 * int5 + 16 * int6)
            self._integral_values.append(int_val[0])
            self._n_splits.append(n)
            if len(self._integral_values) > 1 and \
                    np.abs(np.diff(self._integral_values[-2:])) < self.eps:
                break
        self.int_value, self.sub_interval_num = int_val[0], n

    def plt_precision(self, exact_int=None, is_show=True):
        """
        随着划分次数的增加，积分值的变换曲线
        :return:
        """
        int_values = np.asarray(self._integral_values, dtype=np.float64)
        print("最终精度：%.15e" % (int_values[-1] - int_values[-2]))
        if is_show:
            plt.figure(figsize=(7, 5))
        if exact_int is None:
            plt.semilogy(self._n_splits, int_values, "o-", lw=1.5, label="$Integral \ Value$")
            plt.semilogy(self._n_splits[-1], int_values[-1], "D", label="$%.10e$" % int_values[-1])
            plt.ylabel("积分近似值", fontdict={"fontsize": 16})
        else:
            plt.semilogy(self._n_splits, np.abs(exact_int - int_values), "o-", lw=1.5,
                         label="$\epsilon, \ n_{increment}=%d$" % self.increment)
            plt.semilogy(self._n_splits[-1], np.abs(exact_int - self.int_value), "D",
                         label="$n=%d, \ \epsilon=%.5e$" % (self._n_splits[-1], np.abs(exact_int - self.int_value)))
            plt.ylabel(r"$\epsilon=\vert I - I^* \vert$", fontdict={"fontsize": 18})
        plt.xlabel("划分区间数$n$", fontdict={"fontsize": 16})

        plt.title(r"自适应复合辛普森二重积分收敛曲线$\vert I_{k+1} - I_k \vert=%.2e$"
                  % np.abs(int_values[-1] - int_values[-2]), fontdict={"fontsize": 16})
        plt.legend(frameon=False, fontsize=18)
        plt.grid(ls=":")
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        if is_show:
            plt.show()
