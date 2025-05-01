# -*- coding: UTF-8 -*-
"""
@file:chebyshev_series_approximation.py
@IDE:PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
from scipy import integrate
from function_approximation_03.utils.orthogonal_poly_utils import OrthogonalPolynomialUtils


class ChebyshevSeriesApproximation(OrthogonalPolynomialUtils):
    """
    切比雪夫级数逼近函数，继承OrthogonalPolynomialUtils
    """

    def fit_approximation(self):
        """
        逼近核心算法，即求解系数和递推项
        :return:
        """
        t = self.fun_transform.free_symbols.pop()
        term = sympy.Matrix.zeros(self.k + 1, 1)
        term[0], term[1] = 1, t  # 初始第一、二项
        coefficient = np.zeros(self.k + 1)  # 存储系数
        # 符号函数构造为lambda函数，以便积分运算
        expr = sympy.lambdify(t, self.fun_transform / sympy.sqrt(1 - t ** 2))
        coefficient[0] = integrate.quad(expr, -1, 1)[0] * 2 / np.pi  # 数值积分
        expr = sympy.lambdify(t, term[1] * self.fun_transform /
                              sympy.sqrt(1 - t ** 2))
        coefficient[1] = integrate.quad(expr, -1, 1)[0] * 2 / np.pi
        self.approximation_poly = coefficient[0] / 2 + coefficient[1] * term[1]  # 多项式的前两项
        # 从第三项开始循环求解
        for i in range(2, self.k + 1):
            term[i] = sympy.simplify(2 * t * term[i - 1] - term[i - 2])
            expr = sympy.lambdify(t, term[i] * self.fun_transform /
                                  sympy.sqrt(1 - t ** 2))
            iv = integrate.quad(expr, -1, 1, full_output=1, points=(-1, 1))[0]
            coefficient[i] = 2 / np.pi * iv  # 系数
            self.approximation_poly += coefficient[i] * term[i]

        self.T_coefficient = [term, coefficient]  # 存储逼近多项式各项和对应系数
        self.approximation_poly = sympy.simplify(self.approximation_poly)
        polynomial = sympy.Poly(self.approximation_poly, t)
        self.poly_coefficient = polynomial.coeffs()
        self.polynomial_orders = polynomial.monoms()
        OrthogonalPolynomialUtils.error_analysis(self)  # 调用父类函数

    def plt_approximate(self, is_show=True, is_fh_marker=False):
        """
        可视化函数和逼近多项式函数
        :return:
        """
        OrthogonalPolynomialUtils.plt_approximation(self, "切比雪夫级数", is_show, is_fh_marker)
