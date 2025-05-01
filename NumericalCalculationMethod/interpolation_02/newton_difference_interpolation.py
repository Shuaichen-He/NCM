# -*- coding: UTF-8 -*-
"""
@file:newton_difference_interpolation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import numpy as np
import sympy
import math
import matplotlib.pyplot as plt
from interpolation_02.utils.interpolation_utils import InterpolationUtils


class NewtonDifferenceInterpolation(InterpolationUtils):
    """
    牛顿差分插值法：向前forward、向后backward
    1、向前差分：通常用于计算插值区间的始点x0附近的函数值。
    2、向后差分：通常用于计算插值区间的终点xn附近的函数值。
    """
    diff_val = None  # 差分
    x_start = None  # 存储向前、向后差分的起点值
    h = None  # 等距步长

    def __init__(self, x, y, diff_method="forward"):
        InterpolationUtils.__init__(self, x, y)  # 调用父类初始化
        self.diff_method = diff_method  # 差分方法
        self.h = InterpolationUtils.check_equidistant(self)  # 判断节点是非等距，并获得等距步长
        self.diff_val = InterpolationUtils.cal_difference(self, self.diff_method)  # 获得差分表

    def fit_interp(self):
        """
        生成牛顿差分多项式
        :return:
        """
        t = sympy.Symbol("t")  # 定义符号变量
        term = t  # 差分项
        if self.diff_method == "forward":
            self.polynomial = self.diff_val[0, 0]  # 常数项取x第一个值
            dv = self.diff_val[0, :]  # 向前差分，只需第一行差分值
            self.x_start = self.x[0]  # 起点值为x的第一个值，用于计算插值点t
            for i in range(1, self.n):
                self.polynomial += dv[i] * term / math.factorial(i)
                term *= (t - i)  # 差分项
        elif self.diff_method == "backward":
            self.polynomial = self.diff_val[-1, 0]  # 常数项取x最后一个值
            dv = self.diff_val[-1, :]  # 向后差分，只需最后一行差分值
            self.x_start = self.x[-1]  # 起点值为x的最后一个值，用于计算插值点t
            for i in range(1, self.n):
                self.polynomial += dv[i] * term / math.factorial(i)
                term *= (t + i)  # 差分项
        else:
            raise AttributeError("仅支持牛顿forward、backward差分插值.")

        # 插值多项式特征
        InterpolationUtils.interpolation_polynomial(self, t)  # 调用父类方法

    def predict_x0(self, x0):
        """
        计算插值点x0的插值
        :param x0: 插值点的x坐标
        :return:
        """
        t0 = (x0 - self.x_start) / self.h  # 求解t
        return InterpolationUtils.predict_x0(self, t0)

    def plt_interpolation(self, x0=None, y0=None, fh=None, is_show=True):
        """
        绘制插值多项式和插值点，由于计算插值点的值需要计算t值，故重写父类方法
        :return:
        """
        if is_show:
            plt.figure(figsize=(7, 5))
        plt.plot(self.x, self.y, "ro", label="$(x_i,y_i)$")  # 离散插值节点
        xi = np.linspace(min(self.x), max(self.x), 100)
        yi_hat = self.predict_x0(xi)  # 调用子类方法，而非父类方法，唯一区别
        plt.plot(xi, yi_hat, "k-", label="$g(x)$曲线")  # 可视化插值多项式
        if x0 is not None and y0 is not None:
            plt.plot(x0, y0, "bs", markersize=6, label="$(x_0, \hat y_0)$")  # 可视化所求插值点
        mse = 0.0  # 均方误差
        if fh is not None:
            plt.plot(xi, fh(xi), "r--", label="$f(x)$曲线")  # 真实函数曲线
            mse = np.mean((fh(xi) - yi_hat) ** 2)  # 均方误差
        plt.legend(frameon=False, fontsize=16)  # 添加图例，并取消外方框
        plt.grid(ls=":")  # 添加主要网格线，且是虚线
        plt.xlabel("$x$", fontdict={"fontsize": 18})  # 横坐标标记，latex修饰
        plt.ylabel("$f(x) \ /\  g(x)$", fontdict={"fontsize": 18})  # 纵坐标标记，latex修饰
        # plt.xlabel("$Speed(km/s)$", fontdict={"fontsize": 18})  # 横坐标标记，latex修饰
        # plt.ylabel("$Distance(m)$", fontdict={"fontsize": 18})  # 纵坐标标记，latex修饰
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        if mse != 0.0:
            plt.title("牛顿（$%s$）差分插值：$MSE=%.5e$" % (self.diff_method, mse), fontdict={"fontsize": 18})  # 标题
        else:
            plt.title("牛顿（$%s$）差分插值多项式曲线及插值节点" % self.diff_method, fontdict={"fontsize": 18})  # 标题
        if is_show:
            plt.show()
