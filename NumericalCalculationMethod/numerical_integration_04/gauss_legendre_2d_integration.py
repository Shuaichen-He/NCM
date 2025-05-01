# -*- coding: UTF-8 -*-
"""
@file:gauss_legendre_2d_integration.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import numpy as np
import sympy
import math


class GaussLegendreDoubleIntegration:
    """
    高斯勒让德计算二重积分
    """
    int_value = None  # 最终积分值

    def __init__(self, int_fun, x_span, y_span, zeros_num=10):
        self.ax, self.bx = x_span[0], x_span[1]  # x积分区间
        self.ay, self.by = y_span[0], y_span[1]  # y积分区间
        self.int_fun = int_fun  # 被积函数
        self.n = zeros_num  # 零点数

    def cal_2d_int(self):
        """
        二重数值积分
        :return:
        """
        A_k, zero_points = self._cal_Ak_zeros_()  # 获取插值系数和高斯零点
        # 各变量积分区间转换为[-1, 1]
        A_k_x = A_k * (self.bx - self.ax) / 2
        A_k_y = A_k * (self.by - self.ay) / 2
        zero_points_x = (self.bx - self.ax) / 2 * zero_points + (self.bx + self.ax) / 2
        zero_points_y = (self.by - self.ay) / 2 * zero_points + (self.by + self.ay) / 2
        xy = np.meshgrid(zero_points_x, zero_points_y)
        f_val = self.int_fun(xy[0], xy[1])
        self.int_value = np.dot(np.dot(A_k_x, f_val), A_k_y)
        return self.int_value

    def _cal_Ak_zeros_(self):
        """
        求解勒让德的高斯零点和Ak系数
        :return:
        """
        t = sympy.Symbol("t")
        # 勒让德多项式构造
        p_n = (t ** 2 - 1) ** self.n / math.factorial(self.n) / 2 ** self.n
        diff_p_n = sympy.diff(p_n, t, self.n)  # n阶导数
        # 求解多项式的全部零点
        zero_points = np.asarray(sympy.solve(diff_p_n, t), dtype=np.float64)
        Ak_poly = sympy.lambdify(t, 2 / (1 - t ** 2) / (sympy.diff(diff_p_n, t, 1)) ** 2)
        A_k = Ak_poly(zero_points)  # 求解Ak系数
        return A_k, zero_points
