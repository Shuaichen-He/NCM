# -*- coding: UTF-8 -*-
"""
@file:lagrange_interpolation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import sympy  # 符号运算库
# 导入插值多项式工具类
from interpolation_02.utils.interpolation_utils import InterpolationUtils


class LagrangeInterpolation(InterpolationUtils):
    """
    拉格朗日插值多项式算法类，共包含三个实例方法，且继承InterpolationUtils
    """

    def __init__(self, x, y):
        InterpolationUtils.__init__(self, x, y)  # 调用父类进行参数初始化
        self.interp_base_fun = []  # 拉格朗日独有的实例属性，存储插值基函数

    def fit_interp(self):
        """
        根据已知插值点，生成拉格朗日插值多项式算法，核心即构造插值基函数
        :return: 插值多项式polynomial符号显示
        """
        t = sympy.Symbol("t")  # 定义符号变量
        self.polynomial = 0.0  # 插值多项式实例化
        for i in range(self.n):  # 针对每个数据点
            base_fun = 1.0  # 插值基函数计算，要求i != j
            for j in range(i):
                base_fun = base_fun * (t - self.x[j]) / (self.x[i] - self.x[j])
            for j in range(i + 1, self.n):
                base_fun = base_fun * (t - self.x[j]) / (self.x[i] - self.x[j])
            self.interp_base_fun.append(sympy.expand(base_fun))  # 存储插值基函数
            self.polynomial += base_fun * self.y[i]  # 插值多项式求和
        # 插值多项式的特征项
        InterpolationUtils.interpolation_polynomial(self, t)  # 调用父类

    def plt_interpolation(self, x0=None, y0=None, fh=None, is_show=True):
        """
        可视化插值多项式和插值点
        """
        params = "$Lagrange$", x0, y0, is_show  # 构成元组，封包
        InterpolationUtils.plt_interpolation(self, params, fh)  # 调用父类方法
