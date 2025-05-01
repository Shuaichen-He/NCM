# -*- coding: UTF-8 -*-
"""
@file_name: cubic_bspline_2_order_differentiation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_differentiation_05.cubic_bspline_differentiation import CubicBSplineDifferentiation
from util_font import *


class CubicBSpline2OrderDifferentiation(CubicBSplineDifferentiation):
    """
    三次均匀B样条方法求解二阶数值微分：仅实现第一种边界条件。继承CubicBSplineDifferentiation一阶微分
    读者可根据第2章B样条插值实现其他边界条件系数的求解。
    """

    def predict_diff_x0(self, x0):
        """
        三次B样条方法求解二阶数值微分核心算法，重写父类实例方法
        :return:
        """
        x0 = np.asarray(x0, dtype=np.float64)  # 求微分点
        self.diff_value = np.zeros(len(x0))  # 存储微分值
        # 以x0为中心向两边等分出n份，等分间距为h
        # 以这些点形成三次样条函数S，求出S的系数，再求x0导数
        k = np.linspace(0, self.node_num - 1, self.node_num)  # x0值前后等分n个值的索引
        for i in range(len(x0)):  # 逐个求解给定值的微分
            xi = x0[i] + (k - self.n) * self.h  # 给定x0值前后等分n个值，共2*n+1个值
            y = self.diff_fun(xi)  # 前后拓展n个点后的函数值
            # 求解两端点处一阶导函数值， 采用五点微分公式
            y_0 = np.array([-25, 48, -36, 16, -3]).dot(y[:5]) / (12 * self.h)
            y_n = np.array([3, -16, 36, -48, 25]).dot(y[-5:]) / (12 * self.h)
            # y_0 = (-25 * y[0] + 48 * y[1] - 36 * y[2] + 16 * y[3] - 3 * y[4]) / (12 * self.h)
            # y_n = (3 * y[-5] - 16 * y[-4] + 36 * y[-3] - 48 * y[-2] + 25 * y[-1]) / (12 * self.h)
            cj = self._cal_complete_bspline_(self.h, 2 * self.n, y, y_0, y_n)  # 求解B样条的系数
            self.diff_value[i] = 6 * (y[self.n] - cj[self.n + 1]) / self.h ** 2  # 求解x0点的二阶导数值
            # self.diff_value[i] = (cj[self.n] - 2 * cj[self.n + 1] +
            #                       cj[self.n + 2]) / self.h ** 2  # 另一种形式
        return self.diff_value

    def plt_2_order_different(self, interval, d2fh=None, x0=None, y0=None, is_show=True, is_fh_marker=False):
        """
        可视化，调用父类实例方法
        :return:
        """
        xi = np.linspace(interval[0], interval[1], 200)  # 等距划分
        y_true = d2fh(xi)  # 原函数一阶导函数值
        y_diff = self.predict_diff_x0(xi)  # 三次样条插值求解离散数据数值微分
        # 可视化
        if is_show:
            plt.figure(figsize=(7, 5))
        mae = np.mean(np.abs(y_true - y_diff))
        print("最大绝对值误差：%.10e" % np.max(np.abs(y_true - y_diff)))
        print("平均绝对值误差：%.10e" % mae)
        plt.plot(xi, y_diff, "r-", lw=2, label="$\hat f^{\prime\prime} (x_k): h=%.2f,\ n=%d$" % (self.h, self.n))
        if is_fh_marker:
            xi = interval[0] + np.random.rand(50) * (interval[1] - interval[0])
            xi = np.array(sorted(xi))  # list-->ndarray，升序排列
            y_true_ = d2fh(xi)
            plt.plot(xi, y_true_, "k*", label="$f^{\prime\prime} (x_k), \ x_k \sim U(a, b)$")
        else:
            plt.plot(xi, y_true, "k--", lw=2, label="$f^{\prime\prime} (x)$")
        if x0 is not None and y0 is not None:
            plt.plot(x0, y0, "bo", label="$(x_i, \hat y_i^{\prime\prime})$")
        plt.legend(frameon=False, fontsize=18, loc="best")  # loc="upper right"
        plt.xlabel(r"$x$", fontdict={"fontsize": 18})
        plt.ylabel(r"$f^{\prime\prime} (x) \ / \ \hat f^{\prime\prime} (x)$", fontdict={"fontsize": 18})
        plt.title("三次均匀$B$样条插值数值微分$(MAE=%.2e)$" % mae, fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        if is_show:
            plt.show()
