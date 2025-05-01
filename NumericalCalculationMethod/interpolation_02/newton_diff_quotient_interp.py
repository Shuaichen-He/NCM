# -*- coding: UTF-8 -*-
"""
@file:newton_diff_quotient_interp.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from interpolation_02.utils.interpolation_utils import InterpolationUtils


class NewtonDifferenceQuotient(InterpolationUtils):
    """
    牛顿差商（均差）插值法，继承InterpolationUtils
    """
    diff_quot = None  # 差商表

    def _diff_quotient_(self):
        """
        计算差商（均差）
        :return:
        """
        diff_quot = np.zeros((self.n, self.n))  # 差商表
        diff_quot[:, 0] = self.y  # 第一列存储原插值数据
        for j in range(1, self.n):  # 按列计算，j列标号
            i = np.arange(j, self.n)  # 第j列第j行表示对角线元素，计算j行以下差商
            diff_quot[i, j] = (diff_quot[i, j - 1] - diff_quot[i - 1, j - 1]) / \
                              (self.x[i] - self.x[i - j])
        return diff_quot

    def fit_interp(self):
        """
        牛顿差商插值多项式的生成
        :return:
        """
        self.diff_quot = self._diff_quotient_()  # 计算差商表
        d_q = np.diag(self.diff_quot)  # 取对角线差商元素参与多项式生成
        t = sympy.Symbol("t")  # 定义符号变量
        self.polynomial = d_q[0]  # 初始为第一个y值
        term_poly = (t - self.x[0])
        for i in range(1, self.n):
            self.polynomial += d_q[i] * term_poly
            term_poly *= (t - self.x[i])

        # 插值多项式特征
        InterpolationUtils.interpolation_polynomial(self, t)

    def plt_interpolation(self, x0=None, y0=None, fh=None, is_show=True):
        """
        可视化插值多项式和插值点
        """
        params = "$Newton$差商", x0, y0, is_show  # 构成元组，封包
        InterpolationUtils.plt_interpolation(self, params, fh=fh)  # 调用父类方法
