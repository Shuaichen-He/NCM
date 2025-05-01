# -*- coding: UTF-8 -*-
"""
@file:gauss_legendre_int.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np


class GaussChebyshevIntegration:
    """
    高斯切比雪夫求积公式
    """
    def __init__(self, int_fun, zeros_num=10, cb_type=1):
        self.int_fun = int_fun  # 被积函数
        self.n = int(zeros_num)  # 切比雪夫公式的零点数
        self.zero_points = None  # 切比雪夫高斯零点
        if cb_type in [1, 2]:
            self.cb_type = cb_type  # 选择第一类切比雪夫和第二类切比雪夫
        else:
            raise ValueError("仅能选择1或2，即第一或第二切比雪夫。")
        self.A_k = None  # 求积系数
        self.int_value = None  # 积分值

    def fit_int(self):
        """
        高斯切比雪夫求积公式，核心算法
        :return:
        """
        self.zero_points = np.zeros(self.n)
        if self.cb_type == 1:  # 第一类切比雪夫
            k_i = np.linspace(0, self.n, self.n + 1, endpoint=True)
            self.zero_points = np.cos(np.pi * (2 * k_i + 1) / (2 * self.n + 2))  # 零点
            self.A_k = np.pi / (self.n + 1)  # 插值型系数
            f_val = self.int_fun(self.zero_points)
            self.int_value = self.A_k * sum(f_val)  # 高斯—切比雪夫求积公式
        else:  # 第二类切比雪夫
            k_i = np.linspace(1, self.n, self.n, endpoint=True)
            self.zero_points = np.cos(k_i * np.pi / (self.n + 1))
            self.A_k = np.pi / (self.n + 1) * np.sin(k_i * np.pi / (self.n + 1)) ** 2
            f_val = self.int_fun(self.zero_points)
            self.int_value = np.dot(self.A_k, f_val)
