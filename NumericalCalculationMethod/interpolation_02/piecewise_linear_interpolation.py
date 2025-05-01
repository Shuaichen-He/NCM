# -*- coding: UTF-8 -*-
"""
@file:piecewise_linear_interpolation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from interpolation_02.utils.piecewise_interp_utils import PiecewiseInterpUtils


class PiecewiseLinearInterpolation(PiecewiseInterpUtils):
    """
    分段线性插值，即每两点之间用一次线性函数，继承PiecewiseInterpUtils
    """

    def fit_interp(self):
        """
        核心算法：生成分段线性插值多项式算法
        :return:
        """
        t = sympy.Symbol("t")  # 定义符号变量
        self.polynomial = dict()  # 插值多项式，字典存储，以区间索引为键，值为线性函数
        self.poly_coefficient = np.zeros((self.n - 1, 2))  # 线性函数，故每个区间系数为2个
        for i in range(self.n - 1):
            h_i = self.x[i + 1] - self.x[i]  # 相邻两个数据点步长
            # 分段线性插值公式
            pi = self.y[i + 1] * (t - self.x[i]) / h_i - \
                 self.y[i] * (t - self.x[i + 1]) / h_i
            # pi = (self.y[i + 1] * (t - self.x[i]) - self.y[i] * (t - self.x[i + 1])) / h_i
            self.polynomial[i] = sympy.sympify(pi)  # 分段插值线性函数
            polynomial = sympy.Poly(self.polynomial[i], t)  # 构造多项式对象（线性函数为一次多项式）
            # 某项系数可能为0，故分别对应阶次存储
            mon = polynomial.monoms()
            for j in range(len(mon)):
                self.poly_coefficient[i, mon[j][0]] = polynomial.coeffs()[j]

    def plt_interpolation(self, x0=None, y0=None, fh=None, is_show=True):
        """
        绘制插值多项式和插值点，调用父类方法
        :return:
        """
        params = "分段线性", x0, y0, is_show
        PiecewiseInterpUtils.plt_interpolation(self, params, fh=fh)
