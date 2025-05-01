# -*- coding: UTF-8 -*-
"""
@file:richardson_extrapolation_differentiation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class RichardsonExtrapolationDifferentiation:
    """
    理查森外推算法求解数值微分
    """
    diff_value = None  # x0对应的微分值

    def __init__(self, diff_fun, step=8, h=1):
        self.diff_fun = diff_fun  # 被微分函数
        self.step = step  # 外推步数
        self.h = h  # 微分步长

    def predict_diff_x0(self, x0):
        """
        理查森外推算法求解数值微分，核心算法
        :return:
        """
        x0 = np.asarray(x0, dtype=np.float64)
        self.diff_value = np.zeros(len(x0))  # 存储微分值
        for k in range(len(x0)):  # 逐个求解给定值的微分
            richardson_table = np.zeros(self.step)  # 外推算法计算存储
            # 1. 求得金字塔的底层值
            for i in range(self.step):
                y1 = self.diff_fun(x0[k] + self.h / (2 ** (i + 1)))
                y2 = self.diff_fun(x0[k] - self.h / (2 ** (i + 1)))
                richardson_table[i] = 2 ** i * (y1 - y2) / self.h
            # 2. 逐层求解金字塔值，并外推到指定步数
            rich_tab = np.copy(richardson_table)  # 用于迭代
            for i in range(1, self.step):
                for j in range(i, self.step):
                    # 按公式求得金字塔的每层值
                    rich_tab[j] = (4 ** i * richardson_table[j] -
                                   richardson_table[j - 1]) / (4 ** i - 1)
                richardson_table = rich_tab  # 不断迭代外推
            self.diff_value[k] = richardson_table[-1]  # 顶层值就是所需导数值
        return self.diff_value

    def plt_differentiation(self, interval, dfh, x0=None, y0=None, is_show=True, is_fh_marker=False):  # 参考B样条函数微分
        """
        可视化，理查德外推算法求解离散数据数值微分
        :return:
        """
        xi = np.linspace(interval[0], interval[1], 200)  # 等距划分
        y_true = dfh(xi)  # 原函数一阶导函数值
        y_diff = self.predict_diff_x0(xi)  # 三次样条插值求解离散数据数值微分
        # 可视化
        if is_show:
            plt.figure(figsize=(7, 5))
        mae = np.mean(np.abs(y_true - y_diff))
        print("最大绝对值误差：%.10e" % np.max(np.abs(y_true - y_diff)))
        print("平均绝对值误差：%.10e" % mae)
        plt.plot(xi, y_diff, "r-", lw=2, label="外推$step=%d$" % self.step)
        if is_fh_marker:
            xi = interval[0] + np.random.rand(50) * (interval[1] - interval[0])
            xi = np.array(sorted(xi))  # list-->ndarray，升序排列
            y_true_ = dfh(xi)
            plt.plot(xi, y_true_, "k*", label="$f^{\prime} (x_k), \ x_k \sim U(a, b)$")
        else:
            plt.plot(xi, y_true, "k--", lw=2, label="$f^{\prime} (x)$")
        if x0 is not None and y0 is not None:
            plt.plot(x0, y0, "bo", label="$(x_i, \hat y_i^{\prime})$")
        plt.legend(frameon=False, fontsize=18)
        plt.xlabel(r"$x$", fontdict={"fontsize": 18})
        plt.ylabel(r"$f^{\prime}(x) \ / \ \hat f^{\prime}(x)$", fontdict={"fontsize": 18})
        plt.title("理查森外推算法数值微分$(MAE=%.2e)$" % mae, fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        if is_show:
            plt.show()
