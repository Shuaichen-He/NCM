# -*- coding: UTF-8 -*-
"""
@file:romberg_acceleration_quad.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import mpmath


class RombergAccelerationQuadrature:
    """
    龙贝格加算法
    """
    def __init__(self, fun, int_interval, accelerate_num=10):
        self.fun = fun  # 被积函数
        if len(int_interval) == 2:
            self.a, self.b = int_interval[0], int_interval[1]  # 积分区间
        else:
            raise ValueError("积分区间参数设置有误，格式[a, b].")
        self.acc_num = accelerate_num
        self.int_value = None  # 最终积分值
        self.Romberg_acc_table = None  # 龙贝格加速表

    def fit_int(self):
        """
        龙贝格加算法公式求解数值积分
        :return:
        """
        # 进行accelerate_num次递推，第1列存储逐次分半梯形公式积分值
        self.Romberg_acc_table = np.zeros((self.acc_num + 1, self.acc_num + 1))
        n, h = 1, self.b - self.a  # 初始划分区间的结点数和步长
        T_before, T_next = 0, (self.fun(self.a) + self.fun(self.b)) * h / 2  # 梯形公式
        self.Romberg_acc_table[0, 0] = T_next
        for i in range(1, self.acc_num + 1):
            n, h = 2 * n, h / 2  # 每次递增区间的结点数为原来的2倍，区间步长减半
            T_before = T_next  # 前后两次积分值的迭代
            xi = np.linspace(self.a, self.b, n + 1)  # 等分2 * n + 1个节点
            # 通过节点索引下标，获取奇数索引节点下标值，每次循环只需计算奇数节点值即可
            idx = np.asarray(np.linspace(0, n, n + 1), dtype=np.int64)
            xi_odd = xi[np.mod(idx, 2) == 1]  # 获取奇数节点
            yi_odd = self.fun(xi_odd)  # 每次只需计算计算奇数节点值
            T_next = T_before / 2 + np.sum(yi_odd) * h  # 逐次分半梯形公式
            self.Romberg_acc_table[i, 0] = T_next
        # 龙贝格外算法
        for i in range(self.acc_num):
            pw = mpmath.power(4, i + 1)
            self.Romberg_acc_table[:self.acc_num - i, i + 1] = \
                (pw * self.Romberg_acc_table[1:self.acc_num + 1 - i, i] -
                 self.Romberg_acc_table[:self.acc_num - i, i]) / (pw - 1)  # 龙贝格外推公式

        self.int_value = self.Romberg_acc_table[0, -1]  # 最终积分值
