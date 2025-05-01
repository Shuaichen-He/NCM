# -*- coding: UTF-8 -*-
"""
@file:discrete_data_3_5_points_differentiation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
# 调用插值中的工具类，判断是否等距节点
from interpolation_02.utils.piecewise_interp_utils import PiecewiseInterpUtils


class DiscreteData_3_5_PointsDifferentiation:
    """
    三点公式法和五点公式法求解离散数据的数值微分
    """
    diff_value = None  # 存储给定点x0的微分值

    def __init__(self, x, y, points_type="three"):
        self.x, self.y = np.asarray(x), np.asarray(y)  # 离散数据
        self.n = len(self.x)  # 节点数
        self.points_type = points_type  # 三点公式法和五点公式法两种情况
        utils = PiecewiseInterpUtils(x, y)  # 实例化对象
        self.h = utils.check_equidistant()  # 判断是否等距，并获取微分步长

    def cal_diff(self):
        """
        求解数值微分
        :return:
        """
        if self.points_type.lower() == "three":
            self.diff_value = self._three_points_formula_()
        elif self.points_type.lower() == "five":
            self.diff_value = self._five_points_formula_()
        else:
            raise ValueError("仅支持三点微分公式three和五点微分公式five")
        return self.diff_value

    def _three_points_formula_(self):
        """
        三点公式求解微分：离散数据
        :return:
        """
        diff_value = np.zeros(self.n)  # 存储微分值
        for k in range(self.n):  # 逐个求解给定值的微分
            idx = list(self.x).index(self.x[k])
            if idx == 0:  # 左端点
                dv = -3 * self.y[0] + 4 * self.y[1] - self.y[2]
            elif idx == len(self.x) - 1:  # 右端点
                dv = 3 * self.y[idx] - 4 * self.y[idx - 1] + self.y[idx - 2]
            else:  # 内部点
                dv = self.y[idx + 1] - self.y[idx - 1]  # 斯特林公式
            diff_value[k] = dv / (2 * self.h)
        return diff_value

    def _five_points_formula_(self):
        """
        五点公式求解微分：离散数据
        :return:
        """
        diff_value = np.zeros(self.n)  # 存储微分值
        for k in range(self.n):  # 逐个求解给定值的微分
            idx = list(self.x).index(self.x[k])
            if idx == 0:  # 左端点
                dv = -25 * self.y[0] + 48 * self.y[1] - 36 * self.y[2] + \
                     16 * self.y[3] - 3 * self.y[4]
            elif idx == 1:  # 第二个点
                dv = -3 * self.y[0] - 10 * self.y[1] + 18 * self.y[2] - \
                     6 * self.y[3] + self.y[4]
            elif idx == len(self.x) - 2:  # 倒数第二个点
                dv = np.dot(np.array([-1, 6, -18, 10, 3]), self.y[idx-3: idx + 2])  # 向量点积形式
            elif idx == len(self.x) - 1:  # 右端点
                dv = np.dot(np.array([3, -16, 36, -48, 25]), self.y[idx - 4: idx + 1])  # 向量点积形式
            else:  # 其他情况点
                dv = self.y[idx - 2] - 8 * self.y[idx - 1] + 8 * self.y[idx + 1] - \
                     self.y[idx + 2]
            diff_value[k] = dv / (12 * self.h)
        return diff_value
