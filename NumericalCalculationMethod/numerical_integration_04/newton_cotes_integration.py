# -*- coding: UTF-8 -*-
"""
@file_name: newton_cotes_integration.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
import math
import mpmath  # 任意精度计算

mpmath.mp.dps = 75  # 设置精度


class NewtonCotesIntegration:
    """
    牛顿科特斯积分法：求解科特斯系数，构造插值型求积公式
    """
    def __init__(self, int_fun, int_interval, interval_num=4):
        """
        :param int_fun: 被积函数
        :param int_interval: 积分区间
        :param interval_num: 划分区间数，即等分区间数
        """
        self.int_fun = int_fun
        if len(int_interval) != 2:
            raise ValueError("积分区间参数设置有误，格式[a, b]")
        self.a, self.b = int_interval[0], int_interval[1]  # 积分区间
        self.n = int(interval_num)  # 等分区间数，默认采用科特斯公式
        self.cotes_coefficient = None  # 科特斯系数
        self.int_value = None  # 积分结果

    def fit_cotes_int(self):
        """
        求解数值积分，计算科特斯系数，构造插值型数值积分
        :return:
        """
        # 1. 计算科特斯系数
        if self.n == 1:
            self.cotes_coefficient = np.array([0.5, 0.5])
        else:
            t = sympy.Symbol("t")
            self.cotes_coefficient = np.zeros(self.n + 1)  # 存储科特斯系数
            # 由于科特斯系数对称，故只计算一半
            for i in range(self.n // 2 + 1):
                c = (-1) ** (self.n - i) / self.n / math.factorial(i) / \
                    math.factorial(self.n - i)
                fun_cotes = sympy.lambdify(t, self._cotes_integration_function_(i, t))  # lambda函数
                self.cotes_coefficient[i] = c * mpmath.quad(fun_cotes, (0, self.n))  # 采用库积分函数
            # 反转科特斯系数填充后一半
            if np.mod(self.n, 2) == 1:
                self.cotes_coefficient[self.n // 2 + 1:] = \
                    self.cotes_coefficient[:self.n // 2 + 1][::-1]
            else:
                self.cotes_coefficient[self.n // 2 + 1:] = \
                    self.cotes_coefficient[:self.n // 2][::-1]

        # 2. 构造插值型数值积分
        int_coefficient = (self.b - self.a) * self.cotes_coefficient
        xi = np.linspace(self.a, self.b, self.n + 1)  # 等分区间各端点值
        y_val = self.int_fun(xi)  # 等分区间各端点的函数值
        self.int_value = np.dot(y_val, int_coefficient)  # 插值型积分公式

    def _cotes_integration_function_(self, k, t):
        """
        根据划分区间数，构造科特斯积分函数
        :return:
        """
        fun_c = 1
        for i in range(self.n + 1):
            if i != k:
                fun_c *= (t - i)
        return sympy.expand(fun_c)
