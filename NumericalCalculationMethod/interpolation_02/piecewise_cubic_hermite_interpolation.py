# -*- coding: UTF-8 -*-
"""
@file:piecewise_cubic_hermite_interpolation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from interpolation_02.utils.piecewise_interp_utils import PiecewiseInterpUtils


class PiecewiseCubicHermiteInterpolation(PiecewiseInterpUtils):
    """
    分段三次埃尔米特插值，即分段2点3次埃尔米特插值
    仅有2个插值节点的带1阶导数的埃尔米特插值函数就是1个3次代数多项式函数
    """

    def __init__(self, x, y, dy):
        PiecewiseInterpUtils.__init__(self, x, y)  # 调用父类方法初始化
        self.dy = np.asarray(dy, dtype=np.float64)  # 给定数据点的一阶导数值
        if len(self.y) != len(self.dy):
            raise ValueError("插值数据(x, y, dy)的维度不匹配！")

    def fit_interp(self):
        """
        核心算法：生成分段三次埃尔米特插值插值多项式算法
        :return:
        """
        t = sympy.Symbol("t")  # 定义符号变量
        self.polynomial = dict()  # 插值多项式，字典存储，以区间索引为键，值为3次多项式
        polynomial = dict()  # 插值多项式对象
        self.poly_coefficient = np.zeros((self.n - 1, 4))  # 每个区间为3次多项式，4个系数
        for i in range(self.n - 1):
            hi = self.x[i + 1] - self.x[i]  # 相邻两个数据点步长
            # 分段2点3次埃尔米特插值多项式
            ti, ti1 = t - self.x[i], t - self.x[i + 1]  # 公式子项
            pi = self.y[i] * (1 + 2 * ti / hi) * (ti1 / hi) ** 2 + \
                 self.y[i + 1] * (1 + 2 * ti1 / (-hi)) * (ti / hi) ** 2 + \
                 self.dy[i] * ti * (ti1 / hi) ** 2 + \
                 self.dy[i + 1] * ti1 * (ti / hi) ** 2
            # pi = self.y[i] * (1 + 2 * (t - self.x[i]) / hi) * ((t - self.x[i + 1]) / hi) ** 2 + \
            #      self.y[i + 1] * (1 + 2 * (t - self.x[i + 1]) / (-hi)) * ((t - self.x[i]) / hi) ** 2 + \
            #      self.dy[i] * (t - self.x[i]) * ((t - self.x[i + 1]) / hi) ** 2 + \
            #      self.dy[i + 1] * (t - self.x[i + 1]) * ((t - self.x[i]) / hi) ** 2
            self.polynomial[i] = sympy.expand(pi)  # 分段3次多项式
            polynomial[i] = sympy.Poly(self.polynomial[i], t)  # 构造多项式对象
            # 某项系数可能为0，故分别对应阶次存储
            mon = polynomial[i].monoms()
            for j in range(len(mon)):
                self.poly_coefficient[i, mon[j][0]] = polynomial[i].coeffs()[j]

    def plt_interpolation(self, x0=None, y0=None, fh=None, is_show=True):
        """
        可视化插值多项式和插值点，调用父类实例方法
        :return:
        """
        params = "分段三次埃尔米特", x0, y0, is_show
        PiecewiseInterpUtils.plt_interpolation(self, params, fh=fh)
