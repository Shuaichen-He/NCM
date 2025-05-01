# -*- coding: UTF-8 -*-
"""
@file:cubic_bspline_differentiation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class CubicBSplineDifferentiation:
    """
    三次样条方法求解数值微分：仅实现第一种边界条件
    读者可根据第2章B样条插值实现其他边界条件系数的求解。
    """
    diff_value = None  # 存储给定点x0的微分值

    def __init__(self, diff_fun, n=5, h=0.1):
        self.diff_fun = diff_fun  # 被求微分函数
        self.n = n  # 以x0为中心向两边等分出n份，默认为5
        self.node_num = 2 * self.n + 1  # 等分节点数，两边展开
        self.h = h  # 微分步长

    def predict_diff_x0(self, x0):
        """
        三次样条方法求解数值微分核心算法
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
            y_0 = np.array([-25, 48, -36, 16, -3]).dot(y[:5]) / 12 / self.h
            y_n = np.array([3, -16, 36, -48, 25]).dot(y[-5:]) / 12 / self.h
            # y_0 = (-25 * y[0] + 48 * y[1] - 36 * y[2] + 16 * y[3] - 3 * y[4]) / (12 * self.h)
            # y_n = (3 * y[-5] - 16 * y[-4] + 36 * y[-3] - 48 * y[-2] + 25 * y[-1]) / (12 * self.h)
            # 求解B样条的系数
            coefficient = self._cal_complete_bspline_(self.h, 2 * self.n, y, y_0, y_n)
            # 求解x0点的导数值
            self.diff_value[i] = (coefficient[self.n + 2] - coefficient[self.n]) / (2 * self.h)
        return self.diff_value

    @staticmethod
    def _cal_complete_bspline_(h, n, y, y_0, y_n):
        """
        求解给定点的B样条系数
        :param h: 步长
        :param n: 节点数
        :param y: 函数值
        :param y_0: 端点处一阶导函数值
        :param y_n: 端点处一阶导函数值
        :return: 样条系数coefficient
        """
        coefficient = np.zeros(n + 3)  # 样条函数系数
        # 1. 构造系数矩阵A
        coefficient_matrix, identity_matrix = np.diag(4 * np.ones(n + 1)), np.eye(n + 1)
        mat_low = np.r_[identity_matrix[1:, :], np.zeros((1, n + 1))]
        mat_up = np.r_[np.zeros((1, n + 1)), identity_matrix[:-1, :]]
        coefficient_matrix = coefficient_matrix + mat_low + mat_up  # 形成系数矩阵
        # 2. 构造右端向量b
        b_vector = np.zeros(n + 1)
        b_vector[1:n] = 6 * y[1:n]
        b_vector[0], b_vector[-1] = 6 * y[0] + 2 * h * y_0, 6 * y[-1] - 2 * h * y_n
        # 3. 求解B样条系数
        coefficient[1:-1] = np.linalg.solve(coefficient_matrix, b_vector)  # 求解系数
        coefficient[0] = coefficient[1] - 2 * h * y_0  # 特殊处理
        coefficient[-1] = coefficient[2] + 2 * h * y_n  # 特殊处理
        return coefficient

    def plt_differentiation(self, interval, dfh=None, x0=None, y0=None, is_show=True, is_fh_marker=False):
        """
        可视化，三次均匀B样条函数求解离散数据数值微分
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
        plt.plot(xi, y_diff, "r-", lw=2, label="数值微分$n=%d$" % self.n)
        if is_fh_marker:
            xi = interval[0] + np.random.rand(50) * (interval[1] - interval[0])
            xi = np.array(sorted(xi))  # list-->ndarray，升序排列
            y_true_ = dfh(xi)
            plt.plot(xi, y_true_, "k*", label="$f^{\prime}(x_k), \ x_k \sim U(a, b)$")
        else:
            plt.plot(xi, y_true, "k--", lw=2, label="$f^{\prime}(x)$")
        if x0 is not None and y0 is not None:
            plt.plot(x0, y0, "bo", label="$(x_i, \hat y_i^{\prime})$")
        plt.legend(frameon=False, fontsize=18)
        plt.xlabel(r"$x$", fontdict={"fontsize": 18})
        plt.ylabel(r"$f^{\prime}(x) \ / \ \hat f^{\prime}(x)$", fontdict={"fontsize": 18})
        plt.title("三次均匀$B$样条插值数值微分$(MAE=%.2e)$" % mae, fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        if is_show:
            plt.show()
