# -*- coding: UTF-8 -*-
"""
@file:cubic_spline_corner_interpolation.py.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import numpy as np
import sympy
from interpolation_02.utils.piecewise_interp_utils import PiecewiseInterpUtils


class CubicSplineCornerInterpolation(PiecewiseInterpUtils):
    """
    三次样条插值。继承PiecewiseInterpEntityUtils父类，三转角法
    1、第一种边界条件：complete，给定边界一阶导数且相等
    2、第二种边界条件：second，给定边界的二阶导数且相等。natural，自然边界条件，边界处二阶导数为0，
    3、第三种边界条件：periodic，当被插值函数是以b-a为周期的周期函数时，则要求S(x)也是周期函数
    """

    def __init__(self, x, y, dy=None, d2y=None, boundary_cond="natural"):
        PiecewiseInterpUtils.__init__(self, x, y)
        self.dy, self.d2y = dy, d2y  # 边界条件，一阶导数和二阶导数
        self.boundary_cond = boundary_cond  # 边界条件

    def fit_interp(self):
        """
        生成三次样条插值多项式
        :return:
        """
        t = sympy.Symbol("t")  # 定义符号变量
        self.polynomial = dict()  # 插值多项式
        self.poly_coefficient = np.zeros((self.n - 1, 4))  # 三次多项式4个系数
        if self.boundary_cond == "complete":
            if self.dy is None:
                raise ValueError("请给定数据点边界条件的一阶导数值.")
            self.dy = np.asarray(self.dy, dtype=np.float64)
            self._complete_spline_(t, self.x, self.y, self.dy)
        elif self.boundary_cond == "second":
            if self.d2y is None:
                raise ValueError("请给定数据点边界条件的二阶导数值.")
            self.d2y = np.asarray(self.d2y, dtype=np.float64)
            self._second_spline_(t, self.x, self.y, self.d2y)
        elif self.boundary_cond == "natural":
            self._natural_spline_(t, self.x, self.y)
        elif self.boundary_cond == "periodic":
            self._periodic_spline_(t, self.x, self.y)
        else:
            raise ValueError("边界条件为complete, second, natural, periodic.")

    def _spline_poly_(self, t, x, m):
        """
        构造三次样条插值多项式
        :param t: 符号变量
        :param x: 已知数据点x坐标值
        :param m: 求解边界条件得到的系数 m
        :return:
        """
        for i in range(self.n - 1):
            hi = x[i + 1] - x[i]  # 相邻两个数据点步长
            ti, ti1 = t - x[i], x[i + 1] - t  # 公式子项
            pi = self.y[i] / hi ** 3 * (2 * ti + hi) * ti1 ** 2 + \
                 self.y[i + 1] / hi ** 3 * (2 * ti1 + hi) * ti ** 2 + \
                 m[i] / hi ** 2 * ti * ti1 ** 2 - \
                 m[i + 1] / hi ** 2 * ti1 * ti ** 2
            self.polynomial[i] = sympy.expand(pi)  # 对插值多项式展开
            polynomial = sympy.Poly(self.polynomial[i], t)  # 多项式对象
            # 某项系数可能为0，故分别对应阶次存储
            mon = polynomial.monoms()
            for j in range(len(mon)):
                self.poly_coefficient[i, mon[j][0]] = polynomial.coeffs()[j]

    def _complete_spline_(self, t, x, y, dy):
        """
        求解第一种边界条件， dy为边界值的一阶导数值
        :return:
        """
        coefficient_mat = np.diag(2 * np.ones(self.n))  # 求解m的系数矩阵
        c_vector = np.zeros(self.n)
        for i in range(1, self.n - 1):
            u = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])  # 分母为两个步长和
            lambda_ = (x[i + 1] - x[i]) / (x[i + 1] - x[i - 1])
            c_vector[i] = 3 * lambda_ * (y[i] - y[i - 1]) / (x[i] - x[i - 1]) + \
                          3 * u * (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            # 形成系数矩阵coefficient_mat及向量c_vector
            coefficient_mat[i, i + 1], coefficient_mat[i, i - 1] = u, lambda_
        c_vector[0], c_vector[-1] = 2 * dy[0], 2 * dy[-1]  # 仅仅需要边界两个值
        m_sol = np.reshape(np.linalg.solve(coefficient_mat, c_vector), -1)  # 解方程组
        self._spline_poly_(t, x, m_sol)

    def _second_spline_(self, t, x, y, d2y):
        """
        求解第二种边界条件，d2y为边界值处的二阶导数值
        :return:
        """
        coefficient_mat = np.diag(2 * np.ones(self.n))  # 求解m的系数矩阵
        coefficient_mat[0, 1], coefficient_mat[-1, -2] = 1, 1
        c_vector = np.zeros(self.n)
        for i in range(1, self.n - 1):
            u = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])  # 分母为两个步长和
            lambda_ = (x[i + 1] - x[i]) / (x[i + 1] - x[i - 1])
            c_vector[i] = 3 * lambda_ * (y[i] - y[i - 1]) / (x[i] - x[i - 1]) + \
                          3 * u * (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            coefficient_mat[i, i + 1], coefficient_mat[i, i - 1] = u, lambda_
        # 仅仅需要边界两个值
        c_vector[0] = 3 * (y[1] - y[0]) / (x[1] - x[0]) - (x[1] - x[0]) * d2y[0] / 2
        c_vector[-1] = 3 * (y[-1] - y[-2]) / (x[-1] - x[-2]) - \
                       (x[-1] - x[-2]) * d2y[-1] / 2
        m_sol = np.reshape(np.linalg.solve(coefficient_mat, c_vector), -1)  # 解方程组
        self._spline_poly_(t, x, m_sol)

    def _natural_spline_(self, t, x, y):
        """
        自然边界条件
        :return:
        """
        d2y = np.array([0, 0])  # 仅仅需要边界两个值，且为0
        self._second_spline_(t, x, y, d2y)

    def _periodic_spline_(self, t, x, y):
        """
        周期边界条件
        :return:
        """
        coefficient_mat = np.diag(2 * np.ones(self.n - 1))
        h0, h1, he = x[1] - x[0], x[2] - x[1], x[-1] - x[-2]
        # 针对A矩阵的第一行和最后一行，单独构造
        coefficient_mat[0, 1] = h0 / (h0 + h1)  # 表示u_1
        coefficient_mat[0, -1] = 1 - coefficient_mat[0, 1]  # 表示lamda_1
        coefficient_mat[-1, 0] = he / (h0 + he)  # 表示u_n
        coefficient_mat[-1, -2] = 1 - coefficient_mat[-1, 0]  # 表示lamda_n
        c_vector = np.zeros(self.n - 1)  # 构造右端向量
        c_vector[-1] = 3 * (he * (y[1] - y[0]) / h0 +
                            h0 * (y[-1] - y[-2]) / he) / (h0 + he)
        for i in range(1, self.n - 1):
            u = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
            lambda_ = (x[i + 1] - x[i]) / (x[i + 1] - x[i - 1])
            c_vector[i - 1] = 3 * lambda_ * (y[i] - y[i - 1]) / (x[i] - x[i - 1]) + \
                              3 * u * (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            if i < self.n - 2:  # A矩阵第一行和最后一行已赋值
                coefficient_mat[i, i + 1], coefficient_mat[i, i - 1] = u, lambda_
        m_sol = np.zeros(self.n)
        m_sol[1:] = np.reshape(np.linalg.solve(coefficient_mat, c_vector), -1)
        m_sol[0] = m_sol[-1]
        self._spline_poly_(t, x, m_sol)

    def plt_interpolation(self, x0=None, y0=None, fh=None, is_show=True):
        """
        绘制插值多项式和插值点
        :return:
        """
        params = "三次样条($%s$) " % self.boundary_cond, x0, y0, is_show
        PiecewiseInterpUtils.plt_interpolation(self, params, fh=fh)
