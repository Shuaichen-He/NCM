# -*- coding: UTF-8 -*-
"""
@file_name: pade_rational_fraction_approximation.py
@IDE: PyCharm   Python:3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
import math
from util_font import *


class PadeRationalFractionApproximation:
    """
    帕德形式的有理分式逼近已知函数
    """

    def __init__(self, fun, order=5):
        self.primitive_fun = fun  # 所逼近的函数，符号定义
        self.order = order  # 帕德有理分式的分母多项式的最高次数
        self.rational_fraction = None  # 逼近的有理分式

    def fit_rational_fraction(self):
        """
        帕德形式的有理分式逼近核心算法
        :return:
        """
        t = self.primitive_fun.free_symbols.pop()
        # 1. 求解分母系数q向量
        A = np.zeros((self.order, self.order))  # 系数矩阵n * n
        # p为分子系数，q为分母系数，b为右端向量
        p, q = np.zeros(self.order + 1), np.zeros(self.order + 1)
        b = np.zeros(self.order)
        a = np.zeros(2 * self.order + 1)  # 生成系数矩阵和右端向量的元素
        a[0] = self.primitive_fun.evalf(subs={t: 0})  # 首元素计算f(0)
        for i in range(1, 2 * self.order + 1):  # 其余2n-1个元素计算
            dy_n = sympy.diff(self.primitive_fun, t, i)  # i阶导数
            a[i] = dy_n.evalf(subs={t: 0}) / math.factorial(i)
        for i in range(self.order):
            A[i, :] = a[i + 1:self.order + i + 1]
            b[i] = - a[self.order + i + 1]
        if np.linalg.det(A) == 0.0:
            raise ValueError("Singular matrix.")
        q[1:] = np.linalg.solve(A, b)[::-1]  # 求解并反转

        # 2. 求解分子系数p向量
        p[0], q[0] = a[0], 1
        for i in range(1, self.order + 1):
            p[i] = np.dot(q[np.arange(0, i + 1)], a[i - np.arange(0, i + 1)])

        # 3. 构造分子和分母多项式，分母和分子的阶m = n = order
        molecule, denominator = p[0], 1  # 分别表示分子和分母多项式初始化
        for i in range(1, self.order + 1):
            molecule += p[i] * t ** i  # 计算分子多项式
            denominator += q[i] * t ** i  # 计算分母多项式

        # 4. 构造有理分式，符号形式
        self.rational_fraction = sympy.expand(molecule) / sympy.expand(denominator)

    def cal_x0(self, x0):
        """
        求解有理分式在给定点的值
        :return:
        """
        t = self.rational_fraction.free_symbols.pop()
        rational_fraction = sympy.lambdify(t, self.rational_fraction)
        return rational_fraction(x0)

    def plt_approximate(self, a, b, is_show=True, is_fh_marker=False):
        """
        可视化函数和有理分式函数，a和b为绘图区间左右端点
        :return:
        """
        t = self.rational_fraction.free_symbols.pop()
        if is_show:
            plt.figure(figsize=(7, 6))
        xi = np.linspace(a, b, 200)
        yi_hat = self.cal_x0(xi)  # 求逼近多项式的值
        plt.plot(xi, yi_hat, "r-", lw=2, label="$R_{n,m}(k=%d)$" % self.order)
        fun_expr = sympy.lambdify(t, self.primitive_fun)
        y_true = fun_expr(xi)  # 计算原函数值
        if is_fh_marker:
            xi = a + np.random.rand(100) * (b - a)
            xi = np.array(sorted(xi))  # list-->ndarray，升序排列
            y_true_ = fun_expr(xi)
            plt.plot(xi, y_true_, "k*", label="$f(x_k), \ x_k \sim U(a, b)$")
        else:
            plt.plot(xi, y_true, "k-.", lw=2, label="$f(x)$")

        plt.xlabel(r"$x$", fontdict={"fontsize": 18})
        plt.ylabel(r"$f(x) \ / \ R_{n,m}$", fontdict={"fontsize": 18})
        mae = np.mean(np.abs(yi_hat - y_true))
        print("平均绝对值误差：%.10e" % mae)
        print("最大绝对值误差：%.10e" % np.max(np.abs(yi_hat - y_true)))
        plt.title("帕德有理分式逼近曲线$(MAE=%.2e)$" % mae, fontdict={"fontsize": 18})
        plt.grid(ls=":")
        plt.legend(frameon=False, fontsize=18)
        plt.tick_params(labelsize=18)  # 刻度字体大小18
        if is_show:
            plt.show()