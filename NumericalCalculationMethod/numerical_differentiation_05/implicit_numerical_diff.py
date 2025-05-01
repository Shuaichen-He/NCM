# -*- coding: UTF-8 -*-
"""
@file_name: implicit_numerical_diff.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
# 调用插值中的实体工具类，判断是否等距节点
from interpolation_02.utils.piecewise_interp_utils import PiecewiseInterpUtils
from direct_solution_linear_equations_06.chasing_method_tridiagonal_matrix \
    import ChasingMethodTridiagonalMatrix  # 采用追赶法求解
# 采用三次样条插值求得任意点的微分值
from interpolation_02.cubic_spline_interpolation import CubicSplineInterpolation


class ImplicitNumericalDifferentiation:
    """
    数值微分的隐式格式
    """
    diff_value = None  # 存储给定点x0的微分值

    def __init__(self, x, y):
        self.x, self.y = np.asarray(x), np.asarray(y)  # 离散数据
        self.n = len(x)  # 离散数据点格式
        if self.n < 5:
            print("求解微分值数量过少，请采用其他显式方法求解.")
            exit(0)
        utils = PiecewiseInterpUtils(x, y)  # 实例化对象
        self.h = utils.check_equidistant()  # 判断是否等距，并获取微分步长

    def fit_diff(self):
        """
        求解隐式格式的数值微分
        :return:
        """
        # 构造主次对角线元素
        m_diag, s_diag = 4 * np.ones(self.n - 2), np.ones(self.n - 3)  # 主次对角线元素
        # 用五点公式求解边界点的一阶导数值
        m_0 = np.dot(np.array([-25, 48, -36, 16, -3]), self.y[:5]) / (12 * self.h)  # 左端点
        m_n = np.dot(np.array([3, -16, 36, -48, 25]), self.y[-5:]) / (12 * self.h)  # 右端点
        d_k = 3 / self.h * (self.y[2:] - self.y[:-2])  # 右端向量
        d_k[0], d_k[-1] = d_k[0] - m_0, d_k[-1] - m_n
        self.diff_value = np.zeros(self.n)  # 存储微分值
        self.diff_value[0], self.diff_value[-1] = m_0, m_n
        # 以下用追赶法求解
        cmtm = ChasingMethodTridiagonalMatrix(s_diag, m_diag, s_diag, d_k)
        self.diff_value[1:-1] = cmtm.fit_solve()
        return self.diff_value

    def predict_x0(self, x0):
        """
        求解指定点的微分
        :param x0: 给定求解指定点的ndarray数组
        :return:
        """
        if self.diff_value is None:
            self.fit_diff()
        # 以下采用三次样条插值求得
        csi = CubicSplineInterpolation(self.x, self.diff_value, boundary_cond="natural")
        csi.fit_interp()
        return csi.predict_x0(x0)
