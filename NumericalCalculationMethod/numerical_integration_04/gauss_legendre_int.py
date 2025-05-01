# -*- coding: UTF-8 -*-
"""
@file:gauss_legendre_int.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
import math


class GaussLegendreIntegration:
    """
    高斯—勒让德求积公式
    """
    def __init__(self, int_fun, int_interval, zeros_num=10):
        self.int_fun = int_fun  # 被积函数
        if len(int_interval) == 2:
            self.a, self.b = int_interval[0], int_interval[1]  # 积分区间
        else:
            raise ValueError("积分区间参数设置有误，格式[a, b].")
        self.n = int(zeros_num)  # 勒让德公式的零点数
        self.zero_points, self.A_k = None, None  # 勒让德高斯零点和求积系数A_k
        self.int_value = None  # 积分值

    def fit_int(self):
        """
        高斯勒让德求积公式，核心算法
        :return:
        """
        self._cal_Ak_coef()  # 求解插值型高斯公式系数Ak
        f_val = self.int_fun(self.zero_points)  # 零点函数值
        self.int_value = np.dot(f_val, self.A_k)  # 插值型求积公式
        return self.int_value

    def _cal_zero_points(self):
        """
        求解勒让德的高斯零点
        :return:
        """
        t = sympy.Symbol("t")
        # 勒让德多项式构造
        p_n = (t ** 2 - 1) ** self.n / math.factorial(self.n) / 2 ** self.n
        diff_p_n = sympy.diff(p_n, t, self.n)  # n阶导数
        # 求解多项式的全部零点
        self.zero_points = np.asarray(sympy.solve(diff_p_n, t), dtype=np.float64)
        return diff_p_n, t

    def _cal_Ak_coef(self):
        """
        求解Ak系数
        :return:
        """
        diff_p_n, t = self._cal_zero_points()  # 求解高斯零点
        Ak_poly = sympy.lambdify(t, 2 / (1 - t ** 2) / (sympy.diff(diff_p_n, t, 1)) ** 2)
        self.A_k = Ak_poly(self.zero_points)  # 求解Ak系数
        self.A_k = self.A_k * (self.b - self.a) / 2  # 区间[a, b]转换为积分区间[-1, 1]
        self.zero_points = (self.b - self.a) / 2 * self.zero_points + \
                           (self.b + self.a) / 2  # 区间转换
