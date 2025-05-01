# -*- coding: UTF-8 -*-
"""
@file_name: middle_point_formula_differentiation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from scipy import optimize  # # 科学计算中的优化函数
# 第13章 数值优化的模拟退火算法
from numerical_optimization_13.simulate_anneal import SimulatedAnnealingOptimization
from util_font import *


class MiddlePointFormulaDifferentiation:
    """
    中点公式法，求解函数的一阶导数值
    """
    diff_value = None  # 存储给定点x0的微分值
    diff_error = None  # 存储给定点x0的微分截断误差

    def __init__(self, diff_fun, h=0.1, is_error=False):
        self.sym_fun = diff_fun  # 被微分函数，符号定义，用于误差分析
        self.fun_expr = self._fun_transform(diff_fun)  # lambda函数，用于求值
        self.h = h  # 微分步长，默认0.1
        self.is_error = is_error  # 是否分析误差

    @staticmethod
    def _fun_transform(fun):
        """
        转换为lambda函数
        :param fun:
        :return:
        """
        t = fun.free_symbols.pop()
        return sympy.lambdify(t, fun, "numpy")

    def fit_diff(self, x0):
        """
        核心算法：中点公式法计算给定点x0的微分值
        :param x0: 要求微分值的点向量
        :return:
        """
        n_x0 = len(x0)  # 待求解微分点的数量
        x0 = np.asarray(x0)  # 转化为ndarray数组，便于计算
        yi_b = self.fun_expr(x0 - self.h)  # f(x-h)值
        yi_n = self.fun_expr(x0 + self.h)  # f(x+h)值
        self.diff_value = (yi_n - yi_b) / (2 * self.h)  # 中点公式
        # 分析误差
        if self.is_error:
            self.diff_error = np.zeros(n_x0)  # 存储每个微分点的误差
            for k in range(n_x0):  # 逐个求解给定值的微分误差
                self.diff_error[k] = self.cal_truncation_error(x0[k])
        return self.diff_value

    def cal_truncation_error(self, x_0):
        """
        截断误差分析
        :return:
        """
        t = self.sym_fun.free_symbols.pop()
        d3_fun = self.sym_fun.diff(t, 3)  # 函数的3阶导数
        a, b = x_0 - self.h, x_0 + self.h  # 分析误差区间
        max_val = self._fun_maximize_(d3_fun, t, a, b)  # 3阶导数在指定区间的最大值
        # 截断误差公式
        return self.h ** 2 / 6 * abs(max_val)

    @staticmethod
    def _fun_maximize_(fun, t, a, b):
        """
        求解函数的最大值
        :param fun: 被积函数的n阶导函数
        :param t: 函数的自变量符号
        :param a, b: 求解最大值的区间范围
        :return:
        """
        fun_expr_max = sympy.lambdify(t, -fun)  # 最大值问题转换为最小值问题
        fun_expr_min = sympy.lambdify(t, fun)  # 最小值问题
        # sol_max = optimize.minimize_scalar(fun_expr_max, bounds=(self.a, self.b), method="Bounded")
        # sol_min = optimize.minimize_scalar(fun_expr_min, bounds=(self.a, self.b), method="Bounded")
        sao_max = SimulatedAnnealingOptimization(fun_expr_max, [[a, b]])  # 最大值
        sol_max = np.abs(sao_max.fit_optimize())
        sao_min = SimulatedAnnealingOptimization(fun_expr_min, [[a, b]])  # 最小值
        sol_min = np.abs(sao_min.fit_optimize())
        return sol_max[0] if sol_max[0] > sol_min[0] else sol_min[0]
        # if np.abs(sol_max[0]) > np.abs(sol_min[0]):
        #     return np.abs(sol_max[0])
        # else:
        #     return np.abs(sol_min[0])

    def plt_differentiation(self, interval, x0=None, y0=None, is_show=True,
                            is_fh_marker=False):
        """
        可视化，随机化指定区间微分节点
        :param is_fh_marker: 真实函数是曲线类型还是marker类型
        :return:
        """
        t = self.sym_fun.free_symbols.pop()
        d1_fun = sympy.lambdify(t, self.sym_fun.diff(t, 1))  # 函数的1阶导数
        xi = np.linspace(interval[0], interval[1], 200)  # 等距划分
        y_true = d1_fun(xi)  # 原函数一阶导函数值
        y_diff = self.fit_diff(xi)  # 中点公式求一阶微分
        # 可视化
        if is_show:
            plt.figure(figsize=(7, 5))
        plt.plot(xi, y_diff, "r-", lw=2, label="中点公式, $h=%.2e$" % self.h)
        if is_fh_marker:
            xi = interval[0] + np.random.rand(50) * (interval[1] - interval[0])
            xi = np.array(sorted(xi))  # list-->ndarray，升序排列
            y_true_ = d1_fun(xi)
            plt.plot(xi, y_true_, "k*", label="$f^{\prime}(x_k), \ x_k \sim U(a, b)$")
        else:
            plt.plot(xi, y_true, "k--", lw=2, label="$f^{\prime}(x)$")
        if x0 is not None and y0 is not None:
            plt.plot(x0, y0, "bo", label="$(x_i, \hat y_i^{\prime})$")
        plt.legend(frameon=False, fontsize=18)
        plt.xlabel(r"$x$", fontdict={"fontsize": 20})
        plt.ylabel(r"$f^{\prime}(x)$", fontdict={"fontsize": 20})
        mae = np.mean(np.abs(y_true - y_diff))
        print("最大绝对值误差：%.10e" % np.max(np.abs(y_true - y_diff)))
        print("平均绝对值误差：%.10e" % mae)
        plt.title("中点公式法$(MAE=%.5e)$" % mae, fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        if is_show:
            plt.show()
