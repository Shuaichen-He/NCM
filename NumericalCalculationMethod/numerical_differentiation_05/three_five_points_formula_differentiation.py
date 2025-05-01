# -*- coding: UTF-8 -*-
"""
@file:three_five_points_formula_differentiation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
# 调用插值中的工具类，判断是否等距节点
from interpolation_02.utils.piecewise_interp_utils import PiecewiseInterpUtils
from util_font import *


class ThreeFivePointsFormulaDifferentiation:
    """
    三点公式法和五点公式法求解数值微分
    """
    diff_value = None  # 存储给定点x0的微分值

    def __init__(self, diff_fun, points_type="three", diff_type="forward", h=0.1):
        self.sym_fun = diff_fun  # 被微分函数，符号定义，可用于误差分析（未设计）
        self.diff_fun = self._fun_transform_(diff_fun)  # 被求微分函数
        self.points_type = points_type  # 三点公式法和五点公式法两种情况
        self.diff_type = diff_type  # 若为三点公式，则此项值包括前插法、后插法和斯特林公式
        self.h = h  # 微分步长

    @staticmethod
    def _fun_transform_(fun):  # 参考中点公式
        """
        转换为lambda函数
        :param fun:
        :return:
        """
        t = fun.free_symbols.pop()
        return sympy.lambdify(t, fun, "numpy")

    def predict_diff_x0(self, x0):
        """
        求解数值微分
        :return:
        """
        if self.points_type.lower() == "three":
            self.diff_value = self._three_points_formula_(x0)
        elif self.points_type.lower() == "five":
            self.diff_value = self._five_points_formula_(x0)
        else:
            raise ValueError("仅支持三点微分公式three和五点微分公式five")
        return self.diff_value

    def _three_points_formula_(self, x0):
        """
        三点公式求解微分
        :return:
        """
        n_x0 = len(x0)
        diff_value = np.zeros(n_x0)  # 存储微分值
        for k in range(n_x0):  # 逐个求解给定值的微分
            # 计算函数值：当前点的前2个和后2个步长的函数值
            y = [self.diff_fun(x0[k] + (i - 2) * self.h) for i in range(5)]
            if self.diff_type == "forward":
                diff_value[k] = (-3 * y[2] + 4 * y[3] - y[4]) / (2 * self.h)  # 三点前插
            elif self.diff_type == "backward":
                diff_value[k] = (3 * y[2] - 4 * y[1] + y[0]) / (2 * self.h)  # 三点后插
            elif self.diff_type == "stirling":
                diff_value[k] = (y[3] - y[1]) / (2 * self.h)  # 斯特林公式
            else:
                raise ValueError("三点公式仅适用于（forward, backward, stirling）.")
        return diff_value

    def _five_points_formula_(self, x0):
        """
        五点公式求解微分，即区间[a, b]等分五个点
        :return:
        """
        n_x0 = len(x0)
        diff_value = np.zeros(n_x0)  # 存储微分值
        for k in range(n_x0):  # 逐个求解给定值的微分
            # 计算函数值：当前点的前4个和后4个步长的函数值
            y = [self.diff_fun(x0[k] + (i - 4) * self.h) for i in range(9)]
            if self.diff_type == "middle":  # 第三个点，即中点x2， 精度最高
                diff_value[k] = (y[2] - 8 * y[3] + 8 * y[5] - y[6]) / (12 * self.h)
            elif self.diff_type == "first":  # 第一个点x0，即区间左端点处
                diff_value[k] = \
                    np.array([-25, 48, -36, 16, -3]).dot(y[4:9]) / 12 / self.h
                # diff_value[k] = (-25 * y[4] + 48 * y[5] - 36 * y[6] + 16 * y[7] - 3 * y[8]) / (12 * self.h)
            elif self.diff_type == "second":  # 第二个点x1
                diff_value[k] = np.array([-3, -10, 18, -6, 1]).dot(y[3:8]) / 12 / self.h
                # diff_value[k] = (-3 * y[3] - 10 * y[4] + 18 * y[5] - 6 * y[6] + y[7]) / (12 * self.h)
            elif self.diff_type == "four":  # 第四个点x3
                diff_value[k] = np.array([-1, 6, -18, 10, 3]).dot(y[1:6]) / 12 / self.h
                # diff_value[k] = (3 * y[5] + 10 * y[4] - 18 * y[3] + 6 * y[2] - y[1]) / (12 * self.h)
            elif self.diff_type == "five":  # 第五个点x4， 即区间右端点处
                diff_value[k] = np.array([3, -16, 36, -48, 25]).dot(y[:5]) / 12 / self.h
                # diff_value[k] = (25 * y[4] - 48 * y[3] + 36 * y[2] - 16 * y[1] + 3 * y[0]) / (12 * self.h)
            else:
                raise ValueError("五点公式仅适合于（first, second, middle, four, five）.")
        return diff_value

    def plt_differentiation(self, interval, x0=None, y0=None, is_show=True, is_fh_marker=False):  # 参考中点公式
        """
        可视化，随机化指定区间微分节点
        :return:
        """
        t = self.sym_fun.free_symbols.pop()
        d1_fun = sympy.lambdify(t, self.sym_fun.diff(t, 1))  # 函数的1阶导数
        xi = np.linspace(interval[0], interval[1], 200)  # 等距划分
        y_true = d1_fun(xi)  # 原函数一阶导函数值
        y_diff = self.predict_diff_x0(xi)  # 中点公式求一阶微分
        # 可视化
        if is_show:
            plt.figure(figsize=(7, 5))
        plt.plot(xi, y_diff, "r-", lw=2, label="一阶数值微分, $h=%.2e$" % (self.h))
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
        plt.xlabel(r"$x$", fontdict={"fontsize": 18})
        plt.ylabel(r"$f^{\prime}(x)$", fontdict={"fontsize": 18})
        mae = np.mean(np.abs(y_true - y_diff))
        print("最大绝对值误差：%.10e" % np.max(np.abs(y_true - y_diff)))
        print("平均绝对值误差：%.10e" % mae)
        if self.diff_type == "five":
            plt.title("五点公式($%s$)$(MAE=%.2e)$" % (self.diff_type, mae), fontdict={"fontsize": 18})
        else:
            plt.title("三点公式($%s$)$(MAE=%.2e)$" % (self.diff_type, mae), fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        if is_show:
            plt.show()
