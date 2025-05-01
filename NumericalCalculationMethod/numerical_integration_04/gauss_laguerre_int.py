# -*- coding: UTF-8 -*-
"""
@file:gauss_legendre_int.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
import math


class GaussLaguerreIntegration:
    """
    高斯-拉盖尔求积公式，积分区间[0, ∞)，权函数exp(-x)
    """
    def __init__(self, int_fun, int_interval, zeros_num=10):
        if int_interval[1] is not np.infty:
            raise ValueError("高斯-拉盖尔积分适合积分区间为[a, +∞)")
        self.int_fun = int_fun
        self.a = int_interval[0]  # 积分下限
        self.n = zeros_num + 1  # 拉盖尔公式的零点数
        self.zero_points, self.A_k = None, None  # 拉盖尔高斯零点和求积系数A_k
        self.int_value = None  # 积分值

    def fit_int(self):
        """
        高斯拉盖尔求积公式，核心算法
        :return:
        """
        t = sympy.Symbol("t")
        # 拉盖尔多项式构造
        p_n = sympy.exp(t) * sympy.diff(t ** self.n * sympy.exp(-t), t, self.n)
        self.zero_points = np.asarray(sympy.solve(p_n, t), dtype=np.float64)  # 拉盖尔多项式的零点
        Ak_poly = sympy.lambdify(t, math.factorial(self.n) ** 2 /
                                 (sympy.diff(p_n, t, 1)) ** 2 / t)
        self.A_k = Ak_poly(self.zero_points)  # 求解Ak系数
        # 区间[a, +∞)转换为积分区间[0, +∞)
        f_val = self.int_fun(self.zero_points + self.a) * np.exp(self.zero_points)  # 带权定义被积函数
        # f_val = self.int_fun(self.zeros_points + self.a) * np.exp(-self.a)  # 不带权定义被积函数
        self.int_value = np.dot(f_val, self.A_k)
