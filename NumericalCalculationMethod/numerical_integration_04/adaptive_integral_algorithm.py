# -*- coding: UTF-8 -*-
"""
@file: adaptive_integral_algorithm.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np


class AdaptiveIntegralAlgorithm:
    """
    根据精度要求，自适应积分算法，每个小区间采用辛普生公式求解
    """

    def __init__(self, int_fun, int_interval, eps=1e-8):
        self.int_fun = int_fun  # 被积函数
        if len(int_interval) == 2:
            self.a, self.b = int_interval[0], int_interval[1]  # 积分区间
        else:
            raise ValueError("积分区间参数设置有误，格式[a, b].")
        self.eps = eps  # 积分精度
        self.int_value = None  # 最终积分值
        self.x_node = [self.a, self.b]  # 最终划分的节点分布情况

    def fit_int(self):
        """
        自适应积分算法，采用递归调用格式
        :return:
        """
        self.int_value = self._sub_fit_int(self.a, self.b, self.eps)
        self.x_node = np.asarray(sorted(self.x_node))
        return self.int_value

    def _sub_fit_int(self, a, b, eps):
        """
        递归计算每个子区间的积分值，每个小区间采用辛普生公式求解，
        并根据精度判断各子区间划分前后的积分误差精度
        :return:
        """
        complete_int_value = self._simpson_int_(a, b)  # 整个区间积分值
        mid = (a + b) / 2  # 子区间中点
        left_half = self._simpson_int_(a, mid)  # 左半区间积分值
        right_half = self._simpson_int_(mid, b)  # 右半区间积分值
        # 精度判断
        if abs(complete_int_value - (left_half + right_half)) < 5 * eps:
            int_value = left_half + right_half
        else:
            self.x_node.append(mid)  # 增加划分的节点
            # 不满足精度要求，递归调用
            int_value = self._sub_fit_int(a, mid, eps) + self._sub_fit_int(mid, b, eps)
        return int_value

    def _simpson_int_(self, a, b):
        """
        实现辛普森积分公式
        :param a,b: 子区间的左右端点
        :return:
        """
        mv = self.int_fun((a + b) / 2)  # 中点函数值
        # 辛普森积分公式
        return (b - a) / 6 * (self.int_fun(a) + self.int_fun(b) + 4 * mv)
