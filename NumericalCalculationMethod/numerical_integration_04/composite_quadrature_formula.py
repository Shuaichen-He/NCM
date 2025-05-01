# -*- coding: UTF-8 -*-
"""
@file_name: composite_quadrature_formula.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from scipy import optimize  # # 科学计算中的优化函数
# 第13章 数值优化的模拟退火算法
from numerical_optimization_13.simulate_anneal import SimulatedAnnealingOptimization


class CompositeQuadratureFormula:
    """
    复合求积公式：复合梯形，复合辛普森，复合科特斯
    """

    def __init__(self, int_fun, int_interval, interval_num=16,
                 int_type="simpson", is_remainder=False):
        self.int_fun = int_fun  # 符号函数定义被积分函数
        if len(int_interval) == 2:
            self.a, self.b = int_interval[0], int_interval[1]  # 积分区间
        else:
            raise ValueError("积分区间参数设置有误，格式[a, b].")
        self.n = int(interval_num)  # 等分区间数，默认等分16个区间
        self.int_type = int_type  # 采用复合公式类型：trapz、simpson和cotes
        self.is_remainder = is_remainder  # 是否进行余项分析
        self.int_value = None  # 积分结果
        self.int_remainder = None  # 积分余项

    def fit_int(self):
        """
        复合求积公式，根据int_type选择对应的积分形式
        :return:
        """
        t = self.int_fun.free_symbols.pop()
        fun_expr = sympy.lambdify(t, self.int_fun)  # 转换为lambda函数，便于数值运算
        if self.int_type == "trapezoid":
            self.int_value = self._cal_trapezoid_(t, fun_expr)
        elif self.int_type == "simpson":
            self.int_value = self._cal_simpson_(t, fun_expr)
        elif self.int_type == "cotes":
            self.int_value = self._cal_cotes_(t, fun_expr)
        else:
            raise ValueError("Integration type can only be trapezoid, simpson or cotes!")
        return self.int_value

    def _cal_trapezoid_(self, t, fun_expr):
        """
        复合梯形公式
        :return: int_value积分值
        """
        h = (self.b - self.a) / self.n  # 划分区间步长
        x_k = np.linspace(self.a, self.b, self.n + 1)  # 共n+1个节点
        f_val = fun_expr(x_k)  # 函数值
        int_value = h / 2 * (f_val[0] + f_val[-1] + 2 * sum(f_val[1:-1]))  # 复合梯形公式
        if self.is_remainder:  # 余项分析
            diff_fun = self.int_fun.diff(t, 2)  # 被积函数的2阶导数
            max_val = self._fun_maximize_(diff_fun, t)  # 求函数的最大值
            self.int_remainder = (self.b - self.a) / 12 * h ** 2 * max_val
        return int_value

    def _cal_simpson_(self, t, fun_expr):
        """
        复合辛普森公式
        :return: int_value积分值
        """
        h = (self.b - self.a) / 2 / self.n  # 划分区间步长
        x_k = np.linspace(self.a, self.b, 2 * self.n + 1)  # 共2n + 1个节点
        f_val = fun_expr(x_k)  # 函数值
        idx = np.linspace(0, 2 * self.n, 2 * self.n + 1, dtype=np.int64)  # 索引下标
        f_val_even = f_val[np.mod(idx, 2) == 0]  # 子区间端点值
        f_val_odd = f_val[np.mod(idx, 2) == 1]  # 子区间中点值
        # 复合辛普森公式
        int_value = h / 3 * (f_val[0] + f_val[-1] + 2 * sum(f_val_even[1:-1]) +
                             4 * sum(f_val_odd))
        if self.is_remainder:  # 余项分析
            diff_fun = self.int_fun.diff(t, 4)  # 被积函数的4阶导数
            max_val = self._fun_maximize_(diff_fun, t)  # 求函数的最大值
            self.int_remainder = (self.b - self.a) / 180 * h ** 4 * max_val
        return int_value

    def _cal_cotes_(self, t, fun_expr):
        """
        复合科特斯公式
        :return: int_value积分值
        """
        h = (self.b - self.a) / 4 / self.n  # 划分区间步长
        x_k = np.linspace(self.a, self.b, 4 * self.n + 1)  # 共4n + 1个节点
        f_val = fun_expr(x_k)  # 函数值
        idx = np.linspace(0, 4 * self.n, 4 * self.n + 1, dtype=np.int64)  # 索引下标
        f_val_0 = f_val[np.mod(idx, 4) == 0]  # 下标为4k， 子区间端点值
        f_val_1 = f_val[np.mod(idx, 4) == 1]  # 下标为4k+1，子区间第一个值
        f_val_2 = f_val[np.mod(idx, 4) == 2]  # 下标为4k+2，子区间第二个值
        f_val_3 = f_val[np.mod(idx, 4) == 3]  # 下标为4k+3，子区间第三个值
        # 复合科特斯公式
        int_value = (7 * (f_val_0[0] + f_val_0[-1]) + 14 * sum(f_val_0[1:-1]) +
                     32 * (sum(f_val_1) + sum(f_val_3)) +
                     12 * sum(f_val_2)) * 2 * h / 45
        if self.is_remainder:  # 余项分析
            diff_fun = self.int_fun.diff(t, 6)  # 被积函数的6阶导数
            max_val = self._fun_maximize_(diff_fun, t)  # 求函数的最大值
            self.int_remainder = (self.b - self.a) / 945 * 2 * h ** 6 * max_val
        return int_value

    def _fun_maximize_(self, fun, t):
        """
        求解函数在指定区间内的最大值（且是绝对值最大），本算法采用模拟退火算法
        :param fun: 被积函数的n阶导函数
        :param t: 函数的自变量符号
        :return:
        """
        fun_expr_max = sympy.lambdify(t, -fun, "numpy")  # 最大值问题转换为最小值问题
        fun_expr_min = sympy.lambdify(t, fun, "numpy")  # 最小值问题
        # sol_max = optimize.minimize_scalar(fun_expr_max, bounds=(self.a, self.b), method="Bounded")
        # sol_min = optimize.minimize_scalar(fun_expr_min, bounds=(self.a, self.b), method="Bounded")
        sao_max = SimulatedAnnealingOptimization(fun_expr_max, [[self.a, self.b]])  # 最大值
        sol_max = sao_max.fit_optimize()  # 注释掉50次迭代后，函数最大改变量的精度判断
        sao_min = SimulatedAnnealingOptimization(fun_expr_min, [[self.a, self.b]])  # 最小值
        sol_min = sao_min.fit_optimize()
        if np.abs(sol_max[0]) > np.abs(sol_min[0]):
            return np.abs(sol_max[0])
        else:
            return np.abs(sol_min[0])
