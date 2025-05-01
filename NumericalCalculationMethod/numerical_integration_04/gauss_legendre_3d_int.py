# -*- coding: UTF-8 -*-
"""
@file_name: gauss_legendre_3d_int.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
import math


class GaussLegendreTripleIntegration:
    """
    高斯——勒让德三重积分：
    1. 求勒让德零点
    2. 求插值系数Ak
    3. 做积分区间变换[a, b]-->[-1,1]
    4. 生成三维网格点，计算被积函数的函数值
    5. 根据公式构造计算三重积分值
    """

    def __init__(self, int_fun, x_span, y_span, z_span, zeros_num=None):
        self.int_fun = int_fun  # 被积函数
        self.ax, self.bx = x_span[0], x_span[1]  # x的积分上下限
        self.ay, self.by = y_span[0], y_span[1]  # y的积分上下限
        self.az, self.bz = z_span[0], z_span[1]  # z的积分上下限
        if zeros_num is None:
            self.n_x, self.n_y, self.n_z = 10, 10, 10
        else:
            if len(zeros_num) != 3:
                raise ValueError("零点数设置格式为[nx, ny, nz].")
            self.n_x, self.n_y, self.n_z = zeros_num[:2]
        self.int_value = None  # 最终积分值

    def cal_3d_int(self):
        """
        采用高斯—勒让德计算三重积分，
        :return:
        """
        A_k_x, zero_points_x = self._cal_Ak_zeros_(self.n_x)  # x 计算勒让德的零点与Ak系数
        A_k_y, zero_points_y = self._cal_Ak_zeros_(self.n_y)  # y 计算勒让德的零点与Ak系数
        A_k_z, zero_points_z = self._cal_Ak_zeros_(self.n_z)  # z 计算勒让德的零点与Ak系数
        # 如下为积分区间变换
        A_k_x = A_k_x * (self.bx - self.ax) / 2
        A_k_y = A_k_y * (self.by - self.ay) / 2
        A_k_z = A_k_z * (self.bz - self.az) / 2
        zero_points_x = (self.bx - self.ax) / 2 * zero_points_x + (self.bx + self.ax) / 2
        zero_points_y = (self.by - self.ay) / 2 * zero_points_y + (self.by + self.ay) / 2
        zero_points_z = (self.bz - self.az) / 2 * zero_points_z + (self.bz + self.az) / 2

        xyz = np.meshgrid(zero_points_x, zero_points_y, zero_points_z) # 生成三维网格点
        f_val = self.int_fun(xyz[0], xyz[1], xyz[2])  # 计算函数值

        # 高斯——勒让德三重积分公式
        self.int_value = 0.0
        for j in range(self.n_y):
            for i in range(self.n_x):
                for k in range(self.n_z):
                    self.int_value += A_k_x[i] * A_k_y[j] * A_k_z[k] * f_val[j, i, k]

        return self.int_value

    @staticmethod
    def _cal_Ak_zeros_(n):
        """
        计算勒让德的零点与Ak系数
        :return:
        """
        t = sympy.Symbol("t")
        # 勒让德多项式构造
        p_n = (t ** 2 - 1) ** n / math.factorial(n) / 2 ** n
        diff_p_n = sympy.diff(p_n, t, n)  # 多项式的n阶导数
        # 求解多项式的全部零点，逐次压缩牛顿法求解多项式的全部零点
        zeros_points = np.asarray(sympy.solve(diff_p_n, t), dtype=np.float)
        Ak_poly = sympy.lambdify(t, 2 / (1 - t ** 2) / (diff_p_n.diff(t, 1) ** 2))
        A_k = Ak_poly(zeros_points)  # 求解Ak系数
        return A_k, zeros_points
