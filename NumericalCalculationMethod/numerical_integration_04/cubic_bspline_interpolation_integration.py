# -*- coding: UTF-8 -*-
"""
@file_name: cubic_bspline_interpolation_integration.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from interpolation_02.utils.interpolation_utils import InterpolationUtils


class CubicBSplineInterpolationIntegration:
    """
    三次B样条函数插值积分法，仅采用自然样条，因为其边界出的二阶导数值相等且为零。
    因为所给离散数据有时并不多，求解其边界的二阶导数值或一阶导数值可能存在较大误差。
    """

    def __init__(self, x, y):
        self.x, self.y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if len(self.x) >= 3:
            if len(self.x) != len(self.y):
                raise ValueError("离散数据点维度不匹配。")
            else:
                self.n = len(self.x) - 1  # 离散数据点个数
                interp_eu = InterpolationUtils(x, y)  # 调用插值工具类
                self.h = interp_eu.check_equidistant()  # 判断是否等距
                self.int_value = None  # 离散积分值
        else:
            raise ValueError("离散数据点不能少于3个。")

    def fit_int(self):
        """
        三次B样条函数插值积分算法
        :return:
        """
        c_spline = self.natural_bspline()  # 获得B样条系数

        # 读者可采用如下调用已实现的B样条插值方法实现
        # bsi = BSplineInterpolation(self.x, self.y, d2y=[0, 0], boundary_cond="natural")
        # bsi.fit_interp()
        # c_spline = bsi.natural_bspline()

        # 三次样条函数插值积分公式
        self.int_value = self.h * ((c_spline[0] + c_spline[-1]) / 24 +
                                   (c_spline[1] + c_spline[-2]) / 2 +
                                   23 / 24 * (c_spline[2] + c_spline[-3]) +
                                   np.sum(c_spline[3:-3]))

    def natural_bspline(self):
        """
        第二种自然边界条件的系数求解
        :return:
        """
        # 1. 根据边界条件构造矩阵系数矩阵和右端向量
        c_spline, b_vector = np.zeros(self.n + 3), np.zeros(self.n - 1)
        coefficient_matrix = np.diag(4 * np.ones(self.n - 1))  # 构造对角线元素
        I = np.eye(self.n - 1)  # 构造单位矩阵
        mat_low = np.r_[I[1:, :], np.zeros((1, self.n - 1))]  # 下三角
        mat_up = np.r_[np.zeros((1, self.n - 1)), I[:-1, :]]  # 上三角
        coefficient_matrix = coefficient_matrix + mat_low + mat_up  # 构造三对角矩阵A
        b_vector[1:-1] = 6 * self.y[2:-2]
        b_vector[0] = 6 * self.y[1] - self.y[0]
        b_vector[-1] = 6 * self.y[-2] - self.y[-1]
        # 2. 求解方程组，并重塑为一向量，得到系数向量
        c_spline[2:-2] = np.reshape(np.linalg.solve(coefficient_matrix, b_vector), -1)
        c_spline[1] = self.y[0]  # 表示c0
        c_spline[0] = 2 * c_spline[1] - c_spline[2]  # 表示c_{-1}
        c_spline[-2] = self.y[-1]  # 表示cn
        c_spline[-1] = 2 * c_spline[-2] - c_spline[-3]  # 表示c_{n+1}
        return c_spline
