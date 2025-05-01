# -*- coding: UTF-8 -*-
"""
@file_name: average_parabolic_interpolation_integral.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np


class AverageParabolicInterpolationIntegral:
    """
    平均抛物插值法求解离散数值积分
    """

    def __init__(self, x, y):
        self.x, self.y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if len(self.x) >= 3:
            if len(self.x) != len(self.y):
                raise ValueError("离散数据点维度不匹配。")
            else:
                self.n = len(self.x)  # 离散数据点个数
                self.h = np.diff(self.x)  # 子区间步长
                self.int_value = None  # 离散积分值
        else:
            raise ValueError("离散数据点不能少于3个。")

    def fit_int(self):
        """
        离散数据积分：平均抛物插值算法
        :return:
        """
        # 1. 计算每个子区间的积分参数值
        lambda_ = self.h[:-1] / self.h[1:]  # n - 2个
        delta = self.h[1:] / self.h[:-1]  # n - 2个
        L = delta ** 2 / (1 + delta) * self.y[:-2] - delta * self.y[1:-1] + \
            delta / (1 + delta) * self.y[2:]
        R = lambda_ / (1 + lambda_) * self.y[:-2] - lambda_ * self.y[1:-1] + \
            lambda_ ** 2 / (1 + lambda_) * self.y[2:]
        # 2. 根据平均抛物插值算法公式，计算离散数据积分
        s_term = 3 * (self.y[1:-2] + self.y[2:-1]) - (L[:-1] + R[1:]) / 2
        self.int_value = self.h[0] / 6 * (3 * self.y[0] + 3 * self.y[1] - R[0]) + \
                         np.dot(self.h[1:-1], s_term) / 6 + \
                         self.h[-1] / 6 * (3 * self.y[-2] + 3 * self.y[-1] - L[-1])
