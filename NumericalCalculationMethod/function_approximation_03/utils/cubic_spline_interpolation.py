# -*- coding: UTF-8 -*-
"""
@file:cubic_spline_interpolation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import numpy as np
from direct_solution_linear_equations_06.chasing_method_tridiagonal_matrix \
    import ChasingMethodTridiagonalMatrix  # 第六章追赶法算法


class CubicSplineNaturalInterpolation:
    """
    三次样条插值，仅实现自然边界条件，因为其边界处二阶导数值为0，数值计算
    """
    m = None  # 求解的系数
    y0 = None  # 所求插值点的值

    def __init__(self, x, y):
        self.x, self.y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if len(self.x) == len(self.y):
            self.n = len(self.x)  # 已知数据点的数量
        else:
            raise ValueError("离散数据点(x, y)的维度不一致.")

    def fit_interp_natural(self):
        """
        生成三次样条插值多项式， 自然边界条件
        :return:
        """
        coeff_mat = np.diag(2 * np.ones(self.n))  # 求解m的系数矩阵
        coeff_mat[0, 1], coeff_mat[-1, -2] = 1, 1
        c_vector = np.zeros(self.n)
        for i in range(1, self.n - 1):
            u = (self.x[i] - self.x[i - 1]) / (self.x[i + 1] - self.x[i - 1])  # 分母为两个步长和
            lambda_ = (self.x[i + 1] - self.x[i]) / (self.x[i + 1] - self.x[i - 1])
            c_vector[i] = 3 * lambda_ * (self.y[i] - self.y[i - 1]) / (self.x[i] - self.x[i - 1]) + \
                          3 * u * (self.y[i + 1] - self.y[i]) / (self.x[i + 1] - self.x[i])
            coeff_mat[i, i + 1], coeff_mat[i, i - 1] = u, lambda_
            # 仅仅需要边界两个值
        c_vector[0] = 3 * (self.y[1] - self.y[0]) / (self.x[1] - self.x[0])
        c_vector[-1] = 3 * (self.y[-1] - self.y[-2]) / (self.x[-1] - self.x[-2])
        # 采用第六章已经实现的追赶法求解：基本的高斯消元法，也可采用doolittle方法
        diag_a, diag_b, diag_c = np.diag(coeff_mat, -1), np.diag(coeff_mat), np.diag(coeff_mat, 1)
        gauss_chasing = ChasingMethodTridiagonalMatrix(diag_a, diag_b, diag_c, c_vector)  # 追赶法
        gauss_chasing.fit_solve()  # 求解方程组
        self.m = gauss_chasing.x
        # self.m = np.reshape(np.linalg.solve(coefficient_mat, c_vector), -1)  # 求解方程组，效率低

    def predict_x0(self, x0):
        """
        计算插值点x0的插值
        :return:
        """
        n = len(self.x)
        x0 = np.asarray(x0, dtype=np.float)  # 类型转换
        n0 = len(x0)  # 插值点数量
        y_0 = np.zeros(n0)  # 存储x0的插值
        # 对每一个插值点x0求解插值
        idx = 0  # 默认第一个多项式
        for i in range(len(x0)):
            # 查找被插值点x0所处的区间段索引idx
            for j in range(1, n - 1):
                if self.x[j] <= x0[i] <= self.x[j + 1] or self.x[j] >= x0[i] >= self.x[j + 1]:
                    idx = j
                    break
            hi, t = self.x[idx + 1] - self.x[idx], x0[i]  # 相邻两个数据点步长
            y_0[i] = self.y[idx] / hi ** 3 * (2 * (t - self.x[idx]) + hi) * (self.x[idx + 1] - t) ** 2 + \
                     self.y[idx + 1] / hi ** 3 * (2 * (self.x[idx + 1] - t) + hi) * (t - self.x[idx]) ** 2 + \
                     self.m[idx] / hi ** 2 * (t - self.x[idx]) * (self.x[idx + 1] - t) ** 2 - \
                     self.m[idx + 1] / hi ** 2 * (self.x[idx + 1] - t) * (t - self.x[idx]) ** 2
        return y_0
