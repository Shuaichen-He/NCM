# -*- coding: UTF-8 -*-
"""
@file:gauss_hermite_int.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
import math


class GaussHermiteIntegration:
    """
    高斯—埃尔米特公式求解数值积分
    """
    def __init__(self, int_fun, int_interval=None, zeros_num=1):
        # 不属于积分区间，或输入做判别
        if int_interval is not None:
            if np.isneginf(int_interval[0]) is False or \
                    np.isposinf(int_interval[1]) is False:
                raise ValueError("高斯-埃尔米特积分适合积分区间为[-∞, +∞]")
        self.int_fun = int_fun
        self.n = zeros_num  # 埃尔米特公式的零点数
        self.zero_points, self.A_k = None, None  # 埃尔米特高斯零点和求积系数A_k
        self.int_value = None  # 积分值

    def cal_int(self):
        """
        高斯—埃尔米特求积公式，核心算法
        :return:
        """
        t = sympy.Symbol("t")
        # 埃尔米特多项式构造
        p_n = (-1) ** self.n * sympy.exp(t ** 2) * \
              sympy.diff(sympy.exp(-t ** 2), t, self.n)
        p_n = sympy.simplify(p_n)
        self.zero_points = np.asarray(sympy.solve(p_n, t), dtype=np.float64)  # 埃尔米特零点
        Ak_poly = math.factorial(self.n) * 2 ** (self.n + 1) * \
                  np.sqrt(np.pi) / (sympy.diff(p_n, t, 1)) ** 2
        self.A_k = sympy.lambdify(t, Ak_poly)(self.zero_points)  # 求解Ak系数
        f_val = self.int_fun(self.zero_points) * np.exp(self.zero_points ** 2)
        self.int_value = np.dot(f_val, self.A_k)  # 高斯—埃尔米特求积公式
