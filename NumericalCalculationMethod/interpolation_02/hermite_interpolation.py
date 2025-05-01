# -*- coding: UTF-8 -*-
"""
@file:hermite_interpolation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import numpy as np
import sympy
from interpolation_02.utils.interpolation_utils import InterpolationUtils


class HermiteInterpolation(InterpolationUtils):
    """
    埃尔米特插值：给定函数值及一阶导数值，继承InterpolationUtils
    """

    def __init__(self, x, y, dy):
        InterpolationUtils.__init__(self, x, y)  # 父类初始化
        self.dy = np.asarray(dy, dtype=np.float64)  # 给定数据点的一阶导数值

    def fit_interp(self):
        """
        生成埃尔米特插值多项式
        :return:
        """
        t = sympy.Symbol("t")  # 定义符号变量
        self.polynomial = 0.0  # 插值多项式实例化
        for i in range(self.n):
            h, a = 1.0, 0.0  # 根据公式2.8计算各项表达式
            for j in range(self.n):
                if j != i:
                    h *= (t - self.x[j]) ** 2 / (self.x[i] - self.x[j]) ** 2
                    a += 1 / (self.x[i] - self.x[j])
            self.polynomial += h * ((self.x[i] - t) *
                                    (2 * a * self.y[i] - self.dy[i]) + self.y[i])
        # 插值多项式特征
        InterpolationUtils.interpolation_polynomial(self, t)  # 调用父类实例方法

    def plt_interpolation(self, x0=None, y0=None, fh=None, is_show=True):
        """
        绘制插值多项式和插值点
        :return:
        """
        params = "$Hermite$", x0, y0, is_show
        InterpolationUtils.plt_interpolation(self, params, fh=fh)  # 调用父类实例方法
