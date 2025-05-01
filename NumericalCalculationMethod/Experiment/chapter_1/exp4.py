# -*- coding: UTF-8 -*-
"""
@file_name: exp4.py
@time: 2023-02-09
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import math
from Experiment.util_font import *  # 导入字体文件


class IterationExp:
    """
    实验内容4，迭代法示例算法，面向对象的程序设计
    """

    def __init__(self, x0, eps=1e-15, max_iter=100):
        self.x0 = x0  # 迭代初始值
        self.eps = eps  # 精度控制
        self.max_iter = max_iter  # 最大迭代次数
        self.approximate_values = []  # 存储迭代逼近过程中的值

    def cal_iter(self):
        """
        迭代过程设计
        :return:
        """
        self.approximate_values.append(self.x0)  # 迭代逼近过程中的值
        x_k, tol, iter_ = self.x0, math.inf, 0  # 初始化，迭代值x_k、逼近精度tol和迭代次数iter_
        # 如下为迭代过程，若满足任何一个条件（精度要求和最大迭代次数），则继续迭代
        while tol >= self.eps and iter_ < self.max_iter:
            x_b = x_k  # x_b为迭代的上一次值
            x_k = 1 / (1 + x_b)  # 迭代公式
            self.approximate_values.append(x_k)  # 存储迭代过程中的近似值
            tol = abs(x_k - x_b)  # 相邻两次迭代的绝对差值为精度，改变量较小时，终止
            iter_ += 1  # 迭代次数加一
        return x_k

    def plt_approximate_processing(self):
        """
        可视化迭代逼近过程中的近似值曲线
        :return:
        """
        plt.figure(figsize=(7, 5))
        plt.plot(self.approximate_values, "ko--",
                 label="$x_k, k=%d$" % (len(self.approximate_values) - 1))  # 可视化迭代过程值
        plt.plot(self.approximate_values[0], "D",
                 label="$x_0=%.f, \ \epsilon=10^{-16}$" % self.approximate_values[0])  # 可视化初值
        plt.plot(len(self.approximate_values) - 1, self.approximate_values[-1], "s",
                 label="$x^* = %.15f$" % self.approximate_values[-1])  # 可视化最终近似值
        plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})  # x轴标记
        plt.ylabel(r"$x_k$", fontdict={"fontsize": 18})  # y轴标记
        tol_ = abs(self.approximate_values[-1] - self.approximate_values[-2])  # 算法终止的精度
        plt.title(r"迭代法示例, 终止精度: $\epsilon=\vert x_{k+1} - x_{k} \vert = %.3e$"
                  % tol_, fontdict={"fontsize": 18})  # 标题
        plt.legend(frameon=False, fontsize=16, loc="best")  # 添加图例
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")  # 添加网格线，且是虚线
        plt.show()
