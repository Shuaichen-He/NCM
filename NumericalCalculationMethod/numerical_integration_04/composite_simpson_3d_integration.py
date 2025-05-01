# -*- coding: UTF-8 -*-
"""
@file:composite_simpson_3d_integration.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import numpy as np
from util_font import *


class CompositeSimpsonTripleIntegration:
    """
    复化辛普生三重积分：每次划分区间数递增，对比两次积分精度，满足精度即可。
    """

    def __init__(self, int_fun, x_span, y_span, z_span, eps=1e-6,
                 max_split=20, increment=10):
        self.int_fun = int_fun  # 被积函数
        self.x_span = np.asarray(x_span, dtype=np.float64)  # x积分区间
        self.y_span = np.asarray(y_span, dtype=np.float64)  # y积分区间
        self.z_span = np.asarray(z_span, dtype=np.float64)  # y积分区间
        self.eps = eps  # 积分精度，为前后两次区间划分积分值的变化
        self.max_split = max_split  # 最大划分次数，每次递增10
        self.increment = increment  # 默认划分区间数为10，增量为10
        # 存储每次积分值和每次划分区间数
        self._integral_values, self._n_splits = [], []
        self.int_value = None  # 最终积分值
        self.sub_interval_num = 0  # 子区间划分数

    def fit_3d_int(self):
        """
        三重数值积分
        :return:
        """
        int_val, n = 0, 0
        start = 6 if self.eps <= 1e-10 else 0  # 起始划分数10 ** (start + 1)
        for i in range(start, self.max_split):
            n = self.increment * (i + 1)  # 划分区间数
            hx = np.diff(self.x_span) / n  # x方向积分步长
            hy = np.diff(self.y_span) / n  # y方向积分步长
            hz = np.diff(self.z_span) / n  # z方向积分步长
            # x、y和z划分节点
            xi = np.linspace(self.x_span[0], self.x_span[1], n + 1, endpoint=True)
            yi = np.linspace(self.y_span[0], self.y_span[1], n + 1, endpoint=True)
            zi = np.linspace(self.z_span[0], self.z_span[1], n + 1, endpoint=True)
            vx, vy, vz = np.meshgrid(xi, yi, zi)  # 三维矩阵
            v01, v02 = vx[:-1, :-1, :-1], vx[1:, 1:, 1:]  # f(xi,,), f(x_{i+1},,)
            v11, v12 = vy[:-1, :-1, :-1], vy[1:, 1:, 1:]  # f(,yi,), f(,y_{i+1},)
            v21, v22 = vz[:-1, :-1, :-1], vz[1:, 1:, 1:]  # f(,,zi), f(,,z_{i+1})
            I1, I2, I3 = 0.0, 0.0, 0.0  # 表示I1, I2, I3
            for v0 in [v01, v02]:
                for v1 in [v11, v12]:
                    for v2 in [v21, v22]:
                        I1 += np.sum(self.int_fun(v0, v1, v2))
            xci = np.divide(v01 + v02, 2)  # x各节点中点, f(x_{i+0.5},,)
            yci = np.divide(v11 + v12, 2)  # y各节点中点, f(,y_{i+0.5},)
            zci = np.divide(v21 + v22, 2)  # z各节点中点, f(,,z_{i+0.5})
            for v1 in [v11, v12]:  # f(x_{i+0.5},,)
                for v2 in [v21, v22]:
                    I2 += np.sum(self.int_fun(xci, v1, v2))
            for v0 in [v01, v02]:  # f(,y_{i+0.5},)
                for v2 in [v21, v22]:
                    I2 += np.sum(self.int_fun(v0, yci, v2))
            for v0 in [v01, v02]:  # f(,,z_{i+0.5})
                for v1 in [v11, v12]:
                    I2 += np.sum(self.int_fun(v0, v1, zci))
            for v2 in [v21, v22]:  # f(x_{i+0.5},y_{i+0.5},)
                I3 += np.sum(self.int_fun(xci, yci, v2))
            for v1 in [v11, v12]:  # f(x_{i+0.5},,z_{i+0.5})
                I3 += np.sum(self.int_fun(xci, v1, zci))
            for v0 in [v01, v02]:  # f(,y_{i+0.5},z_{i+0.5})
                I3 += np.sum(self.int_fun(v0, yci, zci))
            # I4表示为f(x_{i+0.5},y_{i+0.5},z_{i+0.5})函数值相加
            I4 = np.sum(self.int_fun(xci, yci, zci))
            # 复合辛普森三重积分公式
            int_val = hx * hy * hz / 216 * (I1 + 4 * I2 + 16 * I3 + 64 * I4)
            self._integral_values.append(int_val[0])  # 存储近似积分
            self._n_splits.append(n)  # 存储划分区间数
            if i > 6:
                print(self._integral_values)
                print(n, np.abs(self._integral_values[-1] - self._integral_values[-2]))
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
        plt.title(r"自适应复合辛普森三重积分收敛曲线$\vert I_{k+1} - I_k \vert=%.2e$"
                  % np.abs(int_values[-2] - int_values[-1]), fontdict={"fontsize": 16})
        plt.legend(frameon=False, fontsize=18)
        plt.grid(ls=":")
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        if is_show:
            plt.show()
