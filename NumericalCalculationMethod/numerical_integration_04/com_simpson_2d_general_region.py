# -*- coding: UTF-8 -*-
"""
@file:com_simpson_2d_general_region.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import numpy as np
import sympy


class GeneralRegionDoubleIntegration:
    """
    一般区域二重积分，首先使用辛普森公式转换二元函数为一元函数，然后采用一重积分计算方法
    本算法采用复合科特斯求解一元函数积分，也可使用复合梯形或复合辛普森公式。
    """

    def __init__(self, int_fun, x_span, c_x, d_x, interval_num=16):
        self.x_span = x_span  # x积分区间
        self.c_x, self.d_x = c_x, d_x  # y积分区间，上下限为函数，且是符号形式
        # 被积函数符号形式定义，替换y的上下限，构成一元函数
        self.int_fun = self.transformation_int_fun_1d(int_fun)
        # self.int_fun = int_fun
        self.n = interval_num  # 复合辛普森对x变量划分的区间数
        self.int_value = None  # 最终积分值

    def transformation_int_fun_1d(self, int_fun):
        """
        转换二元函数为一元函数，然后利用一元函数进行复合辛普森积分
        :return:
        """
        k_fun = (self.d_x - self.c_x) / 2
        x, y = list(int_fun.free_symbols)  # 获取自由变量，集合转变为列表
        if x is not self.d_x.free_symbols.pop():  # 确定对应的符号变量
            x, y = y, x  # 使得一重积分的符号变量为x
        int_fun_1d_1 = int_fun.subs({y: self.c_x})
        int_fun_1d_3 = int_fun.subs({y: self.d_x})
        int_fun_1d_2 = int_fun.subs({y: self.c_x + k_fun})
        int_fun = k_fun / 3 * (int_fun_1d_1 + 4 * int_fun_1d_2 + int_fun_1d_3)
        print(sympy.simplify(int_fun))
        return sympy.simplify(int_fun)

    def fit_2d_int(self):
        """
        一般区域的辛普森二重数值积分
        :return:
        """
        h = (self.x_span[1] - self.x_span[0]) / 4 / self.n  # 划分区间步长
        x_k = np.linspace(self.x_span[0], self.x_span[1], 4 * self.n + 1)  # 共4n + 1个节点
        x = self.int_fun.free_symbols.pop()  # 获取符号变量
        int_fun_expr = sympy.lambdify(x, self.int_fun, "numpy")
        f_val = int_fun_expr(x_k)
        idx = np.linspace(0, 4 * self.n, 4 * self.n + 1, dtype=np.int64)  # 索引下标
        f_val_0 = f_val[np.mod(idx, 4) == 0]  # 下标为4k， 子区间端点值
        f_val_1 = f_val[np.mod(idx, 4) == 1]  # 下标为4k+1，子区间第一个值
        f_val_2 = f_val[np.mod(idx, 4) == 2]  # 下标为4k+2，子区间第二个值
        f_val_3 = f_val[np.mod(idx, 4) == 3]  # 下标为4k+3，子区间第三个值
        # 复合科特斯公式
        self.int_value = (7 * (f_val_0[0] + f_val_0[-1]) + 14 * sum(f_val_0[1:-1]) +
                          32 * (sum(f_val_1) + sum(f_val_3)) +
                          12 * sum(f_val_2)) * 2 * h / 45
        return self.int_value