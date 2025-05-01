# -*- coding: UTF-8 -*-
"""
@file:five_points_second_order_differentiation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
# 调用第2章数据插值中的工具类，判断是否等距节点
from interpolation_02.utils.piecewise_interp_utils import PiecewiseInterpUtils


class FivePointsSecondOrderDifferentiation:
    """
    多点法求解二阶导数，仅实现五点公式，支持函数二阶微分和离散数据二阶微分
    """
    diff_value = None  # 存储给定点x0的微分值

    def __init__(self, diff_fun=None, x=None, y=None, h=0.05):
        self.diff_fun = diff_fun  # 待微分的函数，如果为None，则采用离散数据形式微分
        self.x, self.y = x, y
        self.h = h  # 微分步长

    def fit_2d_diff(self, x0):
        """
        多点法求解二阶导数，核心算法
        :return:
        """
        x0 = np.asarray(x0, dtype=np.float64)  # 被求值
        self.diff_value = np.zeros(len(x0))  # 存储微分值
        if self.diff_fun is not None:  # 存在待微分函数
            self._cal_diff_fun_value_(x0)
        elif self.x is not None and self.y is not None:  # 离散数据形式
            pieu = PiecewiseInterpUtils(self.x, self.y)
            h = pieu.check_equidistant()  # 等距判断，获取步长
            self._cal_diff_discrete_value_(x0, h)
        return self.diff_value

    def _cal_diff_fun_value_(self, x0):
        """
        函数形式，五点公式二阶导数
        :return:
        """
        for k in range(len(x0)):  # 逐个求解给定值的微分
            idx = np.linspace(0, 4, 5)
            y = self.diff_fun(x0[k] + (idx - 2) * self.h)
            # 五点公式（中点，及当前节点前后各四个）
            self.diff_value[k] = np.array([-1, 16, -30, 16, -1]).dot(y[:5]) / \
                                 (12 * self.h ** 2)
            # self.diff_value[k] = (-y[0] + 16 * y[1] - 30 * y[2] + 16 * y[3] - y[4]) / (12 * self.h ** 2)

    def _cal_diff_discrete_value_(self, x0, h):
        """
        离散数据形式，五点公式二阶导数
        :return:
        """
        for k in range(len(x0)):  # 逐个求解给定值的微分
            idx = list(self.x).index(x0[k])
            if idx == 0:  # 左端点
                dv = np.dot(np.array([35, -104, 114, -56, 11]), self.y[:5])
            elif idx == 1:  # 第二个点
                dv = np.dot(np.array([11, -20, 6, 4, -1]), self.y[:5])
            elif idx == len(self.x) - 2:  # 倒数第二个点
                dv = np.dot(np.array([-1, 4, 6, -20, 11]), self.y[idx - 3:idx + 2])
            elif idx == len(self.x) - 1:  # 右端点
                dv = np.dot(np.array([11, -56, 114, -104, 35]), self.y[idx - 4:idx + 1])
            else:  # 其他情况点
                dv = np.dot(np.array([-1, 16, -30, 16, -1]), self.y[idx - 2:idx + 3])
            self.diff_value[k] = dv / (12 * h ** 2)
