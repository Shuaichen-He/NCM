# -*- coding: UTF-8 -*-
"""
@file_name: trapezoidal_integral.py
@time: 2022-10-31
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class TrapezoidalIntegral:
    """
    通过划分积分区间为若干小区间，以小区间的梯形面积近似曲边梯形面积。
    自适应方法，根据精度计算划分区间数，返回满足精度要求的积分值
    """

    def __init__(self, int_fx, a, b, eps=1e-10, max_split_interval_num=1000):
        self.int_fx = int_fx  # 被积函数，非符号定义
        self.a, self.b = a, b  # 积分上下限
        self.eps = eps  # 积分精度，采用划分前后两次精度的绝对值差判断
        self.max_split_interval_num = max_split_interval_num  # 最大划分区间数
        self.int_value = 0.0  # 满足精度要求的积分值
        self.approximate_values = []  # 自适应过程中的积分近似值

    def cal_trapezoid_int_vectorization(self):
        """
        矢量化计算核心算法：自适应计算划分区间数，对每个小区间计算梯形面积，近似曲边梯形面积
        :return:
        """
        int_val_n = (self.b - self.a) / 2 * (self.int_fx(self.a) + self.int_fx(self.b))  # 梯形面积
        self.approximate_values.append(int_val_n)
        tol, split_num = np.infty, 1  # 初始化，逼近精度tol和区间划分数
        while tol > self.eps and split_num < self.max_split_interval_num:
            int_val_b = int_val_n  # 分别模拟当前划分区间数的积分值和下一次划分区间数的积分值
            split_num *= 2  # 每次增加一倍的划分数量：1、2、4、8、16、...
            h = (self.b - self.a) / split_num  # 小区间步长，等分
            x_k = np.linspace(self.a, self.b, split_num + 1)  # 区间端点，为n + 1
            f_xk = self.int_fx(x_k)  # 区间端点的函数值
            int_val_n = h / 2 * np.sum((f_xk[:-1] + f_xk[1:]))  # 积分值，一维向量相加
            self.approximate_values.append(int_val_n)
            tol = np.abs(int_val_n - int_val_b)
        return int_val_n, split_num

    def cal_trapezoid_int_nvectorization(self):
        """
        非矢量化计算核心算法：自适应计算划分区间数，对每个小区间计算梯形面积，近似曲边梯形面积
        :return:
        """
        int_val_n = (self.b - self.a) / 2 * (self.int_fx(self.a) + self.int_fx(self.b))  # 梯形面积
        self.approximate_values.append(int_val_n)
        tol, split_num = np.infty, 1  # 初始化，逼近精度tol和区间划分数
        while tol > self.eps and split_num < self.max_split_interval_num:
            int_val_b = int_val_n  # 分别模拟当前划分区间数的积分值和下一次划分区间数的积分值
            split_num *= 2  # 每次增加一倍的划分数量：1、2、4、8、16、...
            h = (self.b - self.a) / split_num  # 小区间步长，等分，标量计算
            x_k, f_xk = [], []  # 列表用于存储区间端点x_k和对应的函数值f_xk
            for i in range(split_num + 1):
                x_k.append(self.a + i * h)  # 端点值
                f_xk.append(self.int_fx(x_k[-1]))  # 区间端点的函数值
            int_val_n = 0.0  # 积分值
            for i in range(split_num):
                int_val_n += h / 2 * (f_xk[i] + f_xk[i + 1])  # 积分值，标量相加
            self.approximate_values.append(int_val_n)
            tol = np.abs(int_val_n - int_val_b)
        return int_val_n, split_num

    def plt_approximate_processing(self):
        """
        可视化随着划分区间次数的增加，积分近似值的逼近过程
        :return:
        """
        plt.figure(figsize=(14, 5))
        xi = np.linspace(self.a, self.b, 150)
        yi = self.int_fx(xi)
        plt.subplot(121)
        plt.plot(xi, yi, "k-", lw=1)
        plt.fill_between(xi, yi, color="c", alpha=0.5)
        plt.xlabel(r"$x$", fontdict={"fontsize": 20})
        plt.ylabel(r"$f(x)$", fontdict={"fontsize": 20})
        plt.title("被积函数的积分区域", fontdict={"fontsize": 20})
        plt.tick_params(labelsize=18)  # 刻度字体大小16
        plt.subplot(122)
        tol_ = np.abs(self.approximate_values[-1] - self.approximate_values[-2])
        plt.plot(self.approximate_values, "ko--", markerfacecolor="r", markeredgecolor="r",
                 label="$I^* = %.15f$" % self.approximate_values[-1])
        plt.xlabel("划分区间数", fontdict={"fontsize": 20})
        plt.ylabel("积分近似值", fontdict={"fontsize": 20})
        plt.title("积分近似值的逼近过程：$tol = %.5e$" % tol_, fontdict={"fontsize": 20})
        plt.legend(frameon=False, fontsize=18, loc="best")  # 添加图例
        plt.tick_params(labelsize=18)  # 刻度字体大小16
        ticks, orders = [], np.arange(0, len(self.approximate_values), 2)
        for i in orders:
            ticks.append("$2^{%d}$" % i)
        plt.xticks(orders, ticks)
        plt.grid(ls=":")
        plt.show()
